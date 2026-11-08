"""Analyze one chess position with Queen from the command line.

  .venv/bin/python run_queen.py --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

The model's text streams to stderr as it is written; a JSON result prints to stdout.
For the interactive board UI, run server.py instead. See queen_engine.py for details.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from queen_engine import DEFAULT_MODEL_DIR, QueenEngine


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--fen", required=True)
    p.add_argument("--history", type=Path, help="JSON list of prior FENs, chronological; omit current FEN")
    p.add_argument("--prompt", help="Override the evaluation prompt")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--game-id", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    # 8-bit is omitted: bitsandbytes' int8 cuBLASLt kernel fails on a GTX 1660 with CUDA 13.
    p.add_argument("--quant", choices=["4bit", "none"], default="4bit",
                   help="decoder quantization on CUDA; none = fp16, needs ~9 GB (ignored on CPU)")
    p.add_argument("--quiet", action="store_true", help="don't stream the model's text to stderr")
    args = p.parse_args()

    eng = QueenEngine(args.model_dir, device=args.device, quant=args.quant)
    history = json.loads(args.history.read_text()) if args.history else []
    on_text = None if args.quiet else (lambda d: print(d, end="", file=sys.stderr, flush=True))
    result = eng.analyze(args.fen, history, prompt=args.prompt, temperature=args.temperature,
                         max_tokens=args.max_tokens, seed=args.seed, game_id=args.game_id,
                         on_text=on_text)
    print(f"\n[queen] {result['generated_tokens']} tokens in {result['seconds']}s "
          f"({result['tokens_per_second']} tok/s)", file=sys.stderr)
    result.pop("raw_text")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
