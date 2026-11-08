"""Queen (princeton-nlp/queen_pawn-8) inference on a small GPU or CPU, without vLLM.

The official runner (models/queen_pawn-8/infer.py) needs vLLM, BF16 and ~9 GB of
VRAM. This module reproduces its pipeline on plain Hugging Face transformers:

  FEN -> LC0 encoder (CPU, fp32)            -> (16, 64, 1024) board states
  board states -> 16 cross-attn sublayers   -> hooked before decoder layers 0,2,...,30
  chat prompt  -> SmolLM3 decoder           -> analysis text -> best move

so the 3B decoder can be quantized (4-bit by default) to fit a 6 GB card.
The cross-attention math is ported from models/vllm/flamingo.py (DenseXAttnInfer).

    eng = QueenEngine()                      # loads once (~10 s)
    result = eng.analyze(fen, on_text=print) # streams raw text deltas, returns a dict
"""
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

N_XATTN = 16
N_ENC_SQUARES = 64
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "queen_pawn-8"


class DenseXAttn(nn.Module):
    """Gated cross-attention + FFN sublayer; same parameters/math as the release."""

    def __init__(self, encoder_dim, decoder_dim, n_heads=16, n_kv_heads=16):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = decoder_dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.W_Q = nn.Linear(decoder_dim, n_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(encoder_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(encoder_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(n_heads * self.head_dim, decoder_dim, bias=False)
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ffn = nn.Parameter(torch.zeros(1))
        self.norm_y = nn.LayerNorm(decoder_dim)
        self.norm_x = nn.LayerNorm(encoder_dim)
        self.norm_ffn = nn.LayerNorm(decoder_dim)
        self.ffn = nn.Module()
        self.ffn.fc1 = nn.Linear(decoder_dim, 2 * decoder_dim, bias=False)
        self.ffn.fc2 = nn.Linear(2 * decoder_dim, decoder_dim, bias=False)
        self.k = self.v = None  # (B, H, 64, hd), set per position

    def set_board(self, enc):
        """enc (B, 64, encoder_dim) -> cached cross-attn K/V."""
        x = self.norm_x(enc)
        B = enc.shape[0]
        k = self.W_K(x).view(B, N_ENC_SQUARES, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, N_ENC_SQUARES, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        self.k, self.v = k, v

    def forward(self, y):
        """y (B, T, decoder_dim) -> same shape."""
        B, T, _ = y.shape
        q = self.W_Q(self.norm_y(y)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, self.k, self.v)          # (B, H, T, hd)
        attn = attn.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        y = torch.tanh(self.alpha_attn).to(y.dtype) * self.W_O(attn) + y
        h = self.ffn.fc2(F.relu(self.ffn.fc1(self.norm_ffn(y))).square())
        return torch.tanh(self.alpha_ffn).to(y.dtype) * h + y


def _attach_xattn(decoder, xattn_state, dtype, device, encoder_dim=1024):
    layers = decoder.model.layers
    assert 2 * N_XATTN <= len(layers)
    xattn = nn.ModuleList([DenseXAttn(encoder_dim, decoder.config.hidden_size)
                           for _ in range(N_XATTN)])
    xattn.load_state_dict(xattn_state, strict=True)
    xattn.to(device=device, dtype=dtype).eval()

    def make_hook(layer):
        def hook(module, args, kwargs):
            if args:
                return (layer(args[0]), *args[1:]), kwargs
            kwargs["hidden_states"] = layer(kwargs["hidden_states"])
            return args, kwargs
        return hook

    for i, layer in enumerate(xattn):
        layers[2 * i].register_forward_pre_hook(make_hook(layer), with_kwargs=True)
    return xattn


class _DeltaStreamer:
    """transformers streamer that reports newly decoded raw text (special tokens kept)."""

    def __init__(self, tok, on_text):
        self.tok, self.on_text = tok, on_text
        self.ids, self.text, self.skip_prompt = [], "", True

    def put(self, value):
        if self.skip_prompt:            # first call carries the prompt
            self.skip_prompt = False
            return
        self.ids.extend(value.reshape(-1).tolist())
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("�"):     # incomplete UTF-8 sequence; wait for more
            return
        delta, self.text = text[len(self.text):], text
        if delta:
            self.on_text(delta)

    def end(self):
        pass


class _StopOnEvent:
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kw):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool,
                          device=input_ids.device)


class QueenEngine:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device=None, quant="4bit", log=None):
        self.log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
        self.model_dir = Path(model_dir).resolve()
        if str(self.model_dir) not in sys.path:
            sys.path.insert(0, str(self.model_dir))   # bundled models/ and utils/ packages
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging
        from models.encoder import Lc0Bt4HFModel
        hf_logging.set_verbosity_error()

        self.settings = json.loads((self.model_dir / "inference_config.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Turing/older GPUs lack native BF16, so use FP16 there; CPU uses BF16.
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.bfloat16
        self.quant = quant if self.device.type == "cuda" else "none"
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        t0 = time.time()
        self.encoder = Lc0Bt4HFModel.from_pretrained(str(self.model_dir / "lc0"),
                                                     local_files_only=True).eval()
        self.log(f"[queen] loading decoder on {self.device} (quant={self.quant}, {self.dtype}) ...")
        kw = dict(dtype=self.dtype, low_cpu_mem_usage=True)
        if self.device.type == "cuda":
            kw["device_map"] = {"": 0}
            if self.quant == "4bit":
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.dtype, bnb_4bit_use_double_quant=True)
        self.decoder = AutoModelForCausalLM.from_pretrained(str(self.model_dir), **kw).eval()
        if self.device.type == "cpu":
            self.decoder.to(self.device)
        self.xattn = _attach_xattn(self.decoder,
                                   torch.load(self.model_dir / "xattn.pt", map_location="cpu"),
                                   self.dtype, self.device)
        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.vram_gib = (torch.cuda.memory_allocated() / 2**30) if self.device.type == "cuda" else 0.0
        self.load_seconds = time.time() - t0
        self.log(f"[queen] ready in {self.load_seconds:.0f}s"
                 + (f", VRAM {self.vram_gib:.2f} GiB" if self.vram_gib else ""))

    def info(self):
        return {"device": str(self.device),
                "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
                "quant": self.quant, "dtype": str(self.dtype).replace("torch.", ""),
                "vram_gib": round(self.vram_gib, 2), "prompt": self.settings["prompt"]}

    def stop(self):
        self.stop_event.set()

    @torch.no_grad()
    def analyze(self, fen, history=None, *, prompt=None, temperature=0.6, max_tokens=2048,
                seed=None, game_id=0, on_text=None):
        """Analyze one position. on_text(delta) receives raw model text as it is generated."""
        from utils.lc0_planes import encode_fen_batch
        from utils.utils import encode_planes, turn_tensor
        from utils.translate_helpers import Translator
        from utils.eval_policy import choose_move

        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Invalid chess position")
        history = list(history or [])
        for h in history:
            chess.Board(h)
        prompt = prompt if prompt is not None else self.settings["prompt"]
        seed = self.settings["seed"] if seed is None else seed

        with self.lock:
            self.stop_event.clear()
            # 1. Board -> cross-attention K/V.
            planes = encode_fen_batch([board.fen()], [history])
            enc = encode_planes(self.encoder, planes, torch.float32, pov=True,
                                turn=turn_tensor([board.fen()]))
            enc = enc.to(self.device, self.dtype)
            for i, layer in enumerate(self.xattn):
                layer.set_board(enc[:, i])

            # 2. Generate.
            ids = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"].to(self.device)
            digest = hashlib.blake2b(f"{seed}|{game_id}|{board.ply()}".encode(), digest_size=8).digest()
            request_seed = int.from_bytes(digest, "big") % (2**31 - 1)
            torch.manual_seed(request_seed)
            sample = temperature > 0
            t0 = time.time()
            out = self.decoder.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                do_sample=sample, temperature=temperature if sample else None,
                top_k=20 if sample else None, top_p=0.95 if sample else None,
                eos_token_id=self.im_end, pad_token_id=self.tok.pad_token_id,
                streamer=_DeltaStreamer(self.tok, on_text) if on_text else None,
                stopping_criteria=[_StopOnEvent(self.stop_event)])
            seconds = time.time() - t0
            stopped = self.stop_event.is_set()

        gen = out[0, ids.shape[1]:].tolist()
        finish = "stopped" if stopped else ("stop" if gen and gen[-1] == self.im_end else "length")
        if gen and gen[-1] == self.im_end:
            gen = gen[:-1]

        # 3. Pick the move exactly as the official runner does.
        raw = self.tok.decode(gen, skip_special_tokens=False)
        best, source, critical = choose_move(board.fen(), raw, request_seed, self.settings["move_policy"])
        legal_best = best if best is not None and best in board.legal_moves else None
        return {
            "fen": board.fen(),
            "best_move_uci": legal_best.uci() if legal_best else None,
            "best_move_san": board.san(legal_best) if legal_best else None,
            "move_source": source,
            "raw_text": raw,
            "text": Translator(board.turn).decode_absolute(raw),
            "critical_line": critical,
            "finish_reason": finish,
            "generated_tokens": len(gen),
            "seconds": round(seconds, 1),
            "tokens_per_second": round(len(gen) / max(seconds, 1e-9), 1),
            "settings": {"device": str(self.device), "quant": self.quant,
                         "dtype": str(self.dtype).replace("torch.", ""), "temperature": temperature,
                         "top_k": 20, "top_p": 0.95, "request_seed": request_seed},
        }
