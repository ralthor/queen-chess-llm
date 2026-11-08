// Queen local analysis board.
//
// The model writes in POV tokens (see the legend in index.html). This file
//   1. streams an analysis from server.py (/api/analyze, NDJSON),
//   2. parses the raw token text into prose + move/square/piece chips,
//   3. replays each paragraph's moves with chess.js so every move chip knows the
//      exact position it was played from (hover = that position + arrow),
//   4. drives the summary cards, critical-line stepper, eval bar and autoplay.
import { Chessground } from 'https://cdn.jsdelivr.net/npm/chessground@9.2.1/+esm';
import { Chess } from 'https://cdn.jsdelivr.net/npm/chess.js@1.4.0/+esm';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const FILES = 'abcdefgh';
const GLYPH = { w: { K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙' },
                b: { K: '♚', Q: '♛', R: '♜', B: '♝', N: '♞', P: '♟' } };
const NAME = { P: 'pawn', N: 'knight', B: 'bishop', R: 'rook', Q: 'queen', K: 'king' };
const PROMISING_BRUSHES = ['green', 'blue', 'red'];
const EXPECTED_TOKENS = 750; // typical answer length, for the progress bar
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ state
const state = {
  fens: [START_FEN],      // game positions; fens[i] is after i plies
  sans: [], ucis: [],
  cursor: 0,              // which position is on the board
  orientation: 'white',
  queenSide: '',          // '', 'w' or 'b'
  rawView: false, showNums: false,
  analysis: null,         // { rootFen, raw, result, running, t0, deltas, parsed }
  pinned: null,           // pinned preview { fen, shapes, lastMove, label }
  hovering: false,
  step: { idx: -1, timer: null },
};
const game = () => new Chess(state.fens[state.cursor]);
const currentFen = () => state.fens[state.cursor];

// ------------------------------------------------------------------ board
const cg = Chessground($('board'), {
  fen: START_FEN,
  orientation: 'white',
  movable: { free: false, showDests: true, events: { after: onUserMove } },
  premovable: { enabled: false },
  drawable: { enabled: true, visible: true },
  animation: { enabled: true, duration: 180 },
  highlight: { lastMove: true, check: true },
});

function dests(chess) {
  const m = new Map();
  for (const mv of chess.moves({ verbose: true })) {
    if (!m.has(mv.from)) m.set(mv.from, []);
    m.get(mv.from).push(mv.to);
  }
  return m;
}
const colorOf = (turn) => (turn === 'w' ? 'white' : 'black');

// Show the actual game position (with Queen's arrows when they belong to it).
function showGame() {
  const chess = game();
  const busy = state.analysis?.running;
  const queenToMove = state.queenSide && chess.turn() === state.queenSide;
  const last = state.cursor > 0 ? state.ucis[state.cursor - 1] : null;
  cg.set({
    fen: currentFen(),
    turnColor: colorOf(chess.turn()),
    check: chess.inCheck(),
    lastMove: last ? [last.slice(0, 2), last.slice(2, 4)] : undefined,
    movable: { color: busy || queenToMove || chess.isGameOver() ? undefined : colorOf(chess.turn()),
               dests: dests(chess) },
  });
  cg.setAutoShapes(gameShapes());
  $('preview-badge').hidden = true;
  renderNums(currentFen());
}

function showPreview(view) {
  cg.set({ fen: view.fen, lastMove: view.lastMove, check: false, movable: { color: undefined } });
  cg.setAutoShapes(view.shapes || []);
  $('preview-badge').hidden = false;
  $('preview-text').textContent = view.label || 'Previewing line';
  renderNums(view.fen);
}

function restoreView() {
  if (state.pinned) showPreview(state.pinned);
  else showGame();
}

function pin(view) { state.pinned = view; showPreview(view); }
function unpin() { stopStepper(); state.pinned = null; state.step.idx = -1; updateStepper(); showGame(); markActive(null); }

// Default arrows on the game board: promising moves + best move, if the analysis is for this position.
function gameShapes() {
  const a = state.analysis;
  if (!a || a.rootFen !== currentFen() || !a.parsed) return [];
  const shapes = [];
  a.parsed.promising.forEach((mv, i) => {
    if (mv.legal) shapes.push({ orig: mv.from, dest: mv.to, brush: PROMISING_BRUSHES[i] || 'paleGrey' });
  });
  const best = a.parsed.best;
  if (best?.legal && !shapes.some((s) => s.orig === best.from && s.dest === best.to)) {
    shapes.push({ orig: best.from, dest: best.to, brush: 'green' });
  }
  return shapes;
}

// ------------------------------------------------------------------ game flow
function setPosition(fens, sans, ucis, cursor) {
  state.fens = fens; state.sans = sans; state.ucis = ucis; state.cursor = cursor;
  state.pinned = null; state.step.idx = -1; stopStepper();
  showGame(); renderMoveList(); renderTurn(); updateButtons();
  maybeQueenMoves();
}

function playUci(uci) {
  const chess = game();
  let mv;
  try { mv = chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' }); }
  catch { return false; }
  const keep = state.cursor;
  setPosition([...state.fens.slice(0, keep + 1), chess.fen()],
              [...state.sans.slice(0, keep), mv.san],
              [...state.ucis.slice(0, keep), mv.from + mv.to + (mv.promotion || '')], keep + 1);
  return true;
}

function onUserMove(orig, dest) {
  if (!playUci(orig + dest)) showGame();
}

function goTo(i) {
  if (i < 0 || i >= state.fens.length) return;
  state.cursor = i; state.pinned = null; stopStepper();
  showGame(); renderMoveList(); renderTurn(); updateButtons(); renderAnalysisFor();
}

function maybeQueenMoves() {
  const chess = game();
  if (!state.queenSide || chess.isGameOver() || chess.turn() !== state.queenSide) return;
  if (state.cursor !== state.fens.length - 1) return;
  if (state.analysis?.running) return;
  const a = state.analysis;
  if (a && a.rootFen === currentFen() && a.result) { setTimeout(() => playBest(), 500); return; }
  analyze({ autoplay: true });
}

function playBest() {
  const a = state.analysis;
  if (!a || a.rootFen !== currentFen()) return;
  const uci = a.result?.best_move_uci || (a.parsed?.best?.legal && a.parsed.best.from + a.parsed.best.to);
  if (uci) playUci(uci);
}

// ------------------------------------------------------------------ server
async function loadInfo() {
  try {
    const info = await (await fetch('/api/info')).json();
    const hw = info.gpu ? `${info.gpu.replace('NVIDIA GeForce ', '')} · ${info.quant} · ${info.dtype}` : `CPU · ${info.dtype}`;
    setStatus('ready', `Ready · ${hw}`);
    state.hw = hw;
  } catch { setStatus('error', 'Server not reachable. Is server.py running?'); }
}

function setStatus(kind, text) {
  $('status').className = 'status ' + kind;
  $('status-text').textContent = text;
}

async function analyze({ autoplay = false } = {}) {
  if (state.analysis?.running) return;
  const rootFen = currentFen();
  const history = state.fens.slice(0, state.cursor);
  const a = state.analysis = { rootFen, raw: '', result: null, running: true, t0: performance.now(),
                               deltas: 0, parsed: null, autoplay };
  state.pinned = null; state.step.idx = -1;
  setStatus('busy', 'Queen is thinking…');
  updateButtons(); showGame(); renderAnalysis();
  $('progress').hidden = false;
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen: rootFen, history }),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.type === 'delta') { a.raw += msg.text; a.deltas++; scheduleRender(); }
        else if (msg.type === 'done') { a.result = msg.result; a.raw = msg.result.raw_text; }
        else if (msg.type === 'error') throw new Error(msg.message);
      }
    }
    if (!a.result) throw new Error('stream ended early');
    const r = a.result;
    setStatus('ready', `Done · ${r.generated_tokens} tokens in ${r.seconds}s (${r.tokens_per_second} tok/s) · ${state.hw || ''}`);
  } catch (e) {
    setStatus('error', 'Analysis failed: ' + e.message);
  } finally {
    a.running = false;
    $('progress').hidden = true;
    renderAnalysis(); updateButtons();
    if (state.analysis === a) showGame();
    if (a.autoplay && a.result && a.rootFen === currentFen() && a.result.finish_reason !== 'stopped') {
      setTimeout(() => { if (state.analysis === a && a.rootFen === currentFen()) playBest(); }, 700);
    }
  }
}

// ------------------------------------------------------------------ token parsing
// POV square n (1..64) -> absolute square name. pov = side to move at the root ('w'/'b').
function sqName(n, pov) {
  const i = pov === 'w' ? n - 1 : (n - 1) ^ 56;
  return FILES[i % 8] + (Math.floor(i / 8) + 1);
}
const sideColor = (side, pov) => (side === 'M' ? pov : pov === 'w' ? 'b' : 'w');

// One regex for every token construct, in priority order.
const TOKEN_RE = new RegExp([
  // move: piece from [victim] to [promotion]
  '<PIECE_([MO])([PNBRQK])><SQUARE_(\\d+)>(?:<PIECE_([MO])([PNBRQK])>)?<SQUARE_(\\d+)>(?:<PIECE_([MO])([QRBN])>)?',
  // line: square-square [file|rank|diagonal]
  '<SQUARE_(\\d+)>\\s*[-–—]?\\s*<SQUARE_(\\d+)>(?:\\s*(file|rank|diagonal))?',
  // piece [on square]
  '<PIECE_([MO])([PNBRQK])>(?: on <SQUARE_(\\d+)>)?',
  '<SQUARE_(\\d+)>',
  '<PLAYER>|<OPPONENT>|<EMPTY>|<\\|im_end\\|>',
].join('|'), 'g');

function tokenize(text, pov) {
  const segs = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN_RE)) {
    if (m.index > last) segs.push({ kind: 'text', text: text.slice(last, m.index) });
    last = m.index + m[0].length;
    const raw = m[0];
    if (m[1]) {
      segs.push({ kind: 'move', raw, side: m[1], piece: m[2], color: sideColor(m[1], pov),
                  from: sqName(+m[3], pov), to: sqName(+m[6], pov),
                  victim: m[5] || null, promotion: m[8] ? m[8].toLowerCase() : undefined });
    } else if (m[9]) {
      segs.push({ kind: 'line', raw, a: sqName(+m[9], pov), b: sqName(+m[10], pov), what: m[11] || '' });
    } else if (m[12]) {
      segs.push({ kind: 'piece', raw, side: m[12], piece: m[13], color: sideColor(m[12], pov),
                  sq: m[14] ? sqName(+m[14], pov) : null });
    } else if (m[15]) {
      segs.push({ kind: 'square', raw, sq: sqName(+m[15], pov) });
    } else if (raw === '<PLAYER>' || raw === '<OPPONENT>') {
      const white = (raw === '<PLAYER>') === (pov === 'w');
      segs.push({ kind: 'text', text: white ? 'White' : 'Black' });
    } else if (raw === '<EMPTY>') {
      segs.push({ kind: 'text', text: 'empty' });
    }
  }
  if (last < text.length) segs.push({ kind: 'text', text: text.slice(last) });
  return segs;
}

// Replay a run of move segments from the root, attaching fenBefore/fenAfter/san.
//  - "12." / "12..." before a move pins it to that ply of the line (Queen numbers
//    moves from the root position, so the numbers say where a branch starts).
//  - A bare "..." with no number marks an idea ("preparing ...e5"), not a played
//    move: it is checked against the line but does not advance it.
//  - Otherwise the move continues the line; if illegal there, it is retried from
//    earlier positions in the line (alternatives), back to the root.
function replay(segs, rootFen) {
  const root = new Chess(rootFen);
  const rootPly = (root.moveNumber() - 1) * 2 + (root.turn() === 'b' ? 1 : 0);
  let stack = [rootFen];

  const tryAt = (s, k) => {
    if (k < 0 || k >= stack.length) return null;
    const c = new Chess(stack[k]);
    if (c.turn() !== s.color) return null;
    try {
      const mv = c.move({ from: s.from, to: s.to, promotion: s.promotion || 'q' });
      return { san: mv.san, fenBefore: stack[k], fenAfter: c.fen(), ply: k,
               moveNo: new Chess(stack[k]).moveNumber() };
    } catch { return null; }
  };

  segs.forEach((s, i) => {
    if (s.kind !== 'move') return;
    const before = segs[i - 1]?.kind === 'text' ? segs[i - 1].text : '';
    const num = before.match(/(\d+)\s*(\.\.\.|…|\.)\s*$/);
    const idea = !num && /(\.\.\.|…)\s*$/.test(before);
    s.legal = false; s.idea = idea;

    let hit = null;
    if (num) {
      const k = (+num[1] - 1) * 2 + (num[2] === '.' ? 0 : 1) - rootPly;
      hit = tryAt(s, k);
    }
    if (!hit) {
      for (let k = stack.length - 1; k >= 0 && !hit; k--) hit = tryAt(s, k);
    }
    if (hit) {
      Object.assign(s, { legal: true }, hit);
      if (!idea) stack = [...stack.slice(0, hit.ply + 1), hit.fenAfter];
    }
  });
  return segs;
}

function parseAnalysis(raw, rootFen) {
  const pov = new Chess(rootFen).turn();
  const text = raw.replace(/<\|im_end\|>/g, '').replace(/<[^>\s]*$/, ''); // hide half-streamed token
  const field = (name) => {
    const m = text.match(new RegExp('^' + name + ':\\s*(.*)$', 'm'));
    return m ? m[1].trim() : null;
  };
  const bodyEnd = text.search(/^(BEST_MOVE|CRITICAL_LINE|PROMISING_MOVES|EVALUATION):/m);
  const body = (bodyEnd >= 0 ? text.slice(0, bodyEnd) : text).replace(/^ANALYSIS:\s*/, '');
  const paragraphs = body.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean)
    .map((p) => replay(tokenize(p, pov), rootFen));

  const moveList = (s) => (s ? replay(tokenize(s, pov), rootFen).filter((x) => x.kind === 'move') : []);
  const bestF = field('BEST_MOVE');
  const best = bestF ? moveList(bestF)[0] || null : null;
  const critical = moveList(field('CRITICAL_LINE'));
  // Each promising move is an alternative from the root: replay separately.
  const promising = field('PROMISING_MOVES')
    ? tokenize(field('PROMISING_MOVES'), pov).filter((x) => x.kind === 'move').map((m) => replay([m], rootFen)[0])
    : [];
  const evalText = field('EVALUATION');
  return { pov, paragraphs, best, critical, promising,
           evaluation: evalText ? tokenize(evalText, pov) : null,
           evalScore: evalText ? scoreFromText(tokenize(evalText, pov).map((s) => s.text || '').join(' '), pov) : null,
           streamingBody: bodyEnd < 0 };
}

// "the player holds a small advantage of approximately 0.15 pawns" -> +0.15 (White POV)
function scoreFromText(t, pov) {
  const lower = t.toLowerCase();
  let v = null;
  // Magnitude only: the sign comes from the wording ("player is worse by -0.24" is still worse).
  const num = lower.match(/(\d+(?:\.\d+)?)\s*pawns?/);
  if (/\b(mate|checkmate|mating)\b/.test(lower)) v = 10;
  else if (num) v = parseFloat(num[1]);
  else if (/equal|balanced|level/.test(lower)) v = 0;
  if (v === null) return null;
  const playerWorse = /player[^.]*\b(worse|behind|losing|disadvantage)\b/.test(lower)
    || /opponent[^.]*\b(holds|has|is better|winning|advantage|edge)\b/.test(lower) && !/player[^.]*\b(holds|has)\b/.test(lower);
  if (playerWorse) v = -v;
  return pov === 'w' ? v : -v;
}

// ------------------------------------------------------------------ rendering
let renderQueued = false;
function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; renderAnalysis(); });
}

function renderAnalysisFor() {
  // Called when the board moves: the analysis may or may not belong to this position.
  renderAnalysis();
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function rawTokenEl(raw) {
  // Colour PIECE_M* / PIECE_O* tokens so mine/theirs reads at a glance.
  const span = el('span');
  for (const part of raw.split(/(<[^>]+>)/).filter(Boolean)) {
    const t = el('span', /<PIECE_M/.test(part) ? 'tok-m' : /<PIECE_O/.test(part) ? 'tok-o' : '', part);
    span.append(t);
  }
  return span;
}

function moveChip(mv, { big = false, brush = null } = {}) {
  const c = el('span', `chip move ${mv.side === 'M' ? 'mine' : 'theirs'}${mv.legal ? '' : ' illegal'}${mv.idea ? ' idea' : ''}${big ? ' big' : ''}`);
  if (state.rawView) {
    c.classList.add('raw');
    c.append(rawTokenEl(mv.raw));
  } else {
    if (brush) { const sw = el('span', 'swatch'); sw.style.background = BRUSH_COLORS[brush]; c.append(sw); }
    c.append(el('span', 'glyph', GLYPH[mv.color][mv.piece]));
    c.append(el('span', '', mv.legal ? mv.san : `${mv.from}${mv.victim ? '×' : '–'}${mv.to}`));
  }
  c.title = `${mv.color === 'w' ? 'White' : 'Black'} ${NAME[mv.piece]} ${mv.from}→${mv.to}` +
    (mv.legal ? '' : ' (not legal in this line)') + (mv.idea ? ' · an idea, not part of the line' : '') + `\n${mv.raw}`;
  c._mv = mv;
  c.addEventListener('mouseenter', () => hoverMove(mv, c));
  c.addEventListener('mouseleave', endHover);
  c.addEventListener('click', () => clickMove(mv, c));
  return c;
}
const BRUSH_COLORS = { green: '#15781B', blue: '#003088', red: '#882020' };

function squareChip(seg) {
  const c = el('span', 'chip sq' + (state.rawView ? ' raw' : ''));
  if (state.rawView) c.append(rawTokenEl(seg.raw)); else c.textContent = seg.sq;
  c.title = `${seg.raw} → ${seg.sq}`;
  c.addEventListener('mouseenter', () => hoverShapes([{ orig: seg.sq, brush: 'yellow' }]));
  c.addEventListener('mouseleave', endHover);
  return c;
}

function pieceChip(seg) {
  const c = el('span', `chip ${seg.side === 'M' ? 'mine' : 'theirs'}` + (state.rawView ? ' raw' : ''));
  if (state.rawView) c.append(rawTokenEl(seg.raw));
  else {
    c.append(el('span', 'glyph', GLYPH[seg.color][seg.piece]));
    c.append(el('span', '', seg.sq ? `${NAME[seg.piece]} ${seg.sq}` : `${seg.color === 'w' ? 'white' : 'black'} ${NAME[seg.piece]}`));
  }
  c.title = `${seg.color === 'w' ? 'White' : 'Black'} ${NAME[seg.piece]}${seg.sq ? ' on ' + seg.sq : ''}\n${seg.raw}`;
  c.addEventListener('mouseenter', () => {
    const fen = state.analysis?.rootFen || currentFen();
    const squares = seg.sq ? [seg.sq] : findPieces(fen, seg.color, seg.piece);
    hoverShapes(squares.map((s) => ({ orig: s, brush: seg.side === 'M' ? 'green' : 'red' })));
  });
  c.addEventListener('mouseleave', endHover);
  return c;
}

function lineChip(seg) {
  const c = el('span', 'chip sq' + (state.rawView ? ' raw' : ''));
  if (state.rawView) c.append(rawTokenEl(seg.raw));
  else {
    const label = seg.what === 'file' && seg.a[0] === seg.b[0] ? `${seg.a[0]}-file`
      : seg.what === 'rank' && seg.a[1] === seg.b[1] ? `${seg.a[1]}th rank` : `${seg.a}–${seg.b}${seg.what ? ' ' + seg.what : ''}`;
    c.textContent = label;
  }
  c.title = seg.raw;
  c.addEventListener('mouseenter', () => hoverShapes([{ orig: seg.a, dest: seg.b, brush: 'paleBlue' }]));
  c.addEventListener('mouseleave', endHover);
  return c;
}

function findPieces(fen, color, piece) {
  const out = [];
  const b = new Chess(fen).board();
  b.forEach((row) => row.forEach((p) => { if (p && p.color === color && p.type === piece.toLowerCase()) out.push(p.square); }));
  return out;
}

function renderSegs(container, segs) {
  for (const s of segs) {
    if (s.kind === 'text') container.append(document.createTextNode(s.text));
    else if (s.kind === 'move') container.append(moveChip(s));
    else if (s.kind === 'square') container.append(squareChip(s));
    else if (s.kind === 'piece') container.append(pieceChip(s));
    else if (s.kind === 'line') container.append(lineChip(s));
  }
}

function renderAnalysis() {
  const a = state.analysis;
  const prose = $('prose');
  if (!a) return;
  a.parsed = parseAnalysis(a.raw, a.rootFen);
  const p = a.parsed;
  const stale = a.rootFen !== currentFen();

  // Progress
  if (a.running) {
    const secs = (performance.now() - a.t0) / 1000;
    $('progress-fill').style.width = Math.min(100, (a.deltas / EXPECTED_TOKENS) * 100) + '%';
    $('progress-text').textContent = `${a.deltas} tokens · ${secs.toFixed(0)}s · ${(a.deltas / Math.max(secs, 0.1)).toFixed(1)} tok/s`;
  }

  // Prose. It scrolls on its own: keep the reader's place on every re-render, and
  // follow the text as it streams while they are at (or near) the bottom.
  const atBottom = prose.scrollHeight - prose.scrollTop - prose.clientHeight < 40;
  const keepTop = prose.scrollTop;
  prose.replaceChildren();
  if (stale) {
    const note = el('p', 'muted');
    note.append('This analysis is for another position. ');
    const b = el('button', '', 'Go to it');
    b.onclick = () => { const i = state.fens.indexOf(a.rootFen); if (i >= 0) goTo(i); };
    if (state.fens.includes(a.rootFen)) note.append(b);
    prose.append(note);
  }
  if (!a.raw && a.running) prose.append(el('p', 'muted', 'Encoding the board and starting to write…'));
  p.paragraphs.forEach((segs, i) => {
    const para = el('p');
    renderSegs(para, segs);
    if (a.running && p.streamingBody && i === p.paragraphs.length - 1) para.append(el('span', 'cursor'));
    prose.append(para);
  });
  prose.scrollTop = a.running && atBottom ? prose.scrollHeight : keepTop;

  // Best move
  const best = $('best-move');
  best.replaceChildren();
  if (p.best) {
    best.classList.remove('muted');
    best.append(moveChip(p.best, { big: true }));
    const src = a.result?.move_source;
    $('best-source').textContent = src ? `parsed as ${src.replace(/_/g, ' ')}` : '';
  } else { best.classList.add('muted'); best.textContent = a.running ? '…' : '—'; $('best-source').textContent = ''; }

  // Promising
  const prom = $('promising');
  prom.replaceChildren();
  if (p.promising.length) {
    prom.classList.remove('muted');
    p.promising.forEach((m, i) => prom.append(moveChip(m, { brush: PROMISING_BRUSHES[i] })));
  } else { prom.classList.add('muted'); prom.textContent = a.running ? '…' : '—'; }

  // Critical line
  const crit = $('critical');
  crit.replaceChildren();
  if (p.critical.length) {
    crit.classList.remove('muted');
    p.critical.forEach((m, i) => {
      if (m.legal && (m.color === 'w' || i === 0)) crit.append(el('span', 'num', m.moveNo + (m.color === 'w' ? '.' : '…')));
      const c = moveChip(m);
      c.addEventListener('click', (e) => { e.stopImmediatePropagation(); stopStepper(); stepTo(i); }, { capture: true });
      crit.append(c);
    });
  } else { crit.classList.add('muted'); crit.textContent = a.running ? '…' : '—'; }
  updateStepper();

  // Evaluation
  if (p.evaluation) {
    const v = p.evalScore;
    $('eval-number').classList.remove('muted');
    $('eval-number').textContent = v == null ? '?' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
    if (v != null) $('eval-number').append(el('span', 'eval-side',
      Math.abs(v) < 0.05 ? 'equal' : v > 0 ? 'White better' : 'Black better'));
    const t = $('eval-text'); t.replaceChildren(); renderSegs(t, p.evaluation);
    setEvalBar(v);
  } else {
    $('eval-number').classList.add('muted');
    $('eval-number').textContent = a.running ? '…' : '—';
    $('eval-text').textContent = '';
  }

  renderLegendExample();
  if (!a.running && !state.pinned && !state.hovering) showGame();
}

function setEvalBar(v) {
  const score = v ?? 0;
  const whiteFrac = 1 / (1 + Math.exp(-score * 0.8));
  $('evalbar-white').style.height = (whiteFrac * 100).toFixed(1) + '%';
  $('evalbar-label').textContent = v == null ? '?' : Math.abs(score).toFixed(1);
}

// ------------------------------------------------------------------ hover / click on chips
let activeChip = null;
function markActive(c) {
  activeChip?.classList.remove('active');
  activeChip = c; c?.classList.add('active');
}

function moveView(mv, after) {
  const label = `${mv.moveNo ?? ''}${mv.color === 'w' ? '.' : '…'} ${mv.san || mv.from + mv.to}` +
    (after ? '' : ' (about to be played)');
  return after
    ? { fen: mv.fenAfter, lastMove: [mv.from, mv.to], shapes: [], label: 'After ' + label }
    : { fen: mv.fenBefore, lastMove: undefined, shapes: [{ orig: mv.from, dest: mv.to, brush: mv.side === 'M' ? 'green' : 'red' }], label };
}

function hoverMove(mv) {
  state.hovering = true;
  if (mv.legal) showPreview(moveView(mv, false));
  else hoverShapes([{ orig: mv.from, dest: mv.to, brush: 'paleGrey' }]);
}

function hoverShapes(shapes) {
  state.hovering = true;
  const base = state.pinned ? state.pinned.shapes || [] : gameShapes();
  cg.setAutoShapes([...base, ...shapes]);
}

function endHover() { state.hovering = false; restoreView(); }

function clickMove(mv, chip) {
  if (!mv.legal) return;
  stopStepper();
  state.step.idx = -1; updateStepper();
  markActive(chip);
  pin(moveView(mv, true));
}

// ------------------------------------------------------------------ critical line stepper
function stepTo(i) {
  const line = state.analysis?.parsed?.critical.filter((m) => m.legal) || [];
  if (!line.length) return;
  if (i < 0) { unpin(); return; }
  i = Math.min(i, line.length - 1);
  state.step.idx = i;
  const chips = [...$('critical').querySelectorAll('.chip')];
  markActive(chips[i] || null);
  pin(moveView(line[i], true));
  state.pinned.label = `Critical line · move ${i + 1}/${line.length}: ` + state.pinned.label.replace(/^After /, '');
  $('preview-text').textContent = state.pinned.label;
  updateStepper();
}

function stopStepper() {
  clearInterval(state.step.timer); state.step.timer = null;
  $('step-play').textContent = '▶ Play line';
}

function playLine() {
  if (state.step.timer) { stopStepper(); return; }
  const n = state.analysis?.parsed?.critical.filter((m) => m.legal).length || 0;
  if (!n) return;
  if (state.step.idx >= n - 1) state.step.idx = -1;
  $('step-play').textContent = '⏸ Pause';
  stepTo(state.step.idx + 1);
  state.step.timer = setInterval(() => {
    if (state.step.idx >= n - 1) { stopStepper(); return; }
    stepTo(state.step.idx + 1);
  }, 1000);
}

function updateStepper() {
  const n = state.analysis?.parsed?.critical.filter((m) => m.legal).length || 0;
  for (const id of ['step-first', 'step-prev', 'step-play', 'step-next']) $(id).disabled = !n;
  $('step-pos').textContent = n ? (state.step.idx >= 0 ? `move ${state.step.idx + 1} of ${n}` : `${n} moves`) : '';
}

// ------------------------------------------------------------------ square-number overlay
function renderNums(fen) {
  const box = $('sqnums');
  box.hidden = !state.showNums;
  if (!state.showNums) return;
  const pov = new Chess(fen).turn();
  box.replaceChildren();
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      // screen (row, col) -> absolute square index
      const rank = state.orientation === 'white' ? 7 - row : row;
      const file = state.orientation === 'white' ? col : 7 - col;
      const abs = rank * 8 + file;
      const n = (pov === 'w' ? abs : abs ^ 56) + 1;
      const s = el('span', '', String(n));
      s.title = `<SQUARE_${n}> = ${FILES[file]}${rank + 1} when ${pov === 'w' ? 'White' : 'Black'} is to move`;
      box.append(s);
    }
  }
}

// ------------------------------------------------------------------ legend example
function renderLegendExample() {
  const box = $('legend-example');
  const mv = state.analysis?.parsed?.best;
  box.replaceChildren();
  if (!mv) { box.append(el('span', 'muted', 'After an analysis, its best move is decoded here token by token.')); return; }
  const pov = state.analysis.parsed.pov;
  const toks = mv.raw.match(/<[^>]+>/g);
  const who = (s) => (s === 'M' ? `my (${pov === 'w' ? 'White' : 'Black'})` : `opponent's (${pov === 'w' ? 'Black' : 'White'})`);
  let sqSeen = 0;
  toks.forEach((t, i) => {
    const b = el('div', 'tokbox');
    b.append(el('code', '', t));
    let meaning;
    const pm = t.match(/<PIECE_([MO])([PNBRQK])>/);
    const sm = t.match(/<SQUARE_(\d+)>/);
    if (pm) meaning = `${who(pm[1])} ${NAME[pm[2]]}`;
    else if (sm) { meaning = `${sqSeen++ ? 'to' : 'from'} square ${sm[1]} = ${sqName(+sm[1], pov)}`; }
    b.append(el('div', '', meaning));
    box.append(b);
    if (i < toks.length - 1) box.append(el('span', 'arrow', '·'));
  });
  box.append(el('span', 'arrow', '='));
  const res = el('div', 'tokbox');
  res.append(el('code', '', mv.san || '?'));
  res.append(el('div', '', 'standard notation'));
  box.append(res);
}

// ------------------------------------------------------------------ side panels
function renderMoveList() {
  const box = $('movelist');
  box.replaceChildren();
  const startNo = new Chess(state.fens[0]).moveNumber();
  const blackFirst = new Chess(state.fens[0]).turn() === 'b';
  state.sans.forEach((san, i) => {
    const ply = i + (blackFirst ? 1 : 0);
    if (ply % 2 === 0) box.append(el('span', 'num', `${startNo + Math.floor(ply / 2)}.`));
    else if (i === 0) box.append(el('span', 'num', `${startNo}…`));
    const m = el('span', 'mv' + (state.cursor === i + 1 ? ' current' : ''), san);
    m.onclick = () => goTo(i + 1);
    box.append(m);
  });
  box.scrollTop = box.scrollHeight;
}

function renderTurn() {
  const c = game();
  let t = `${c.turn() === 'w' ? 'White' : 'Black'} to move`;
  if (c.isCheckmate()) t = `Checkmate. ${c.turn() === 'w' ? 'Black' : 'White'} wins.`;
  else if (c.isDraw()) t = 'Draw.';
  else if (state.queenSide && c.turn() === state.queenSide) t += ' (Queen)';
  else if (state.queenSide) t += ' (you)';
  $('turn-line').textContent = t;
}

function updateButtons() {
  const running = state.analysis?.running;
  $('btn-analyze').disabled = !!running || game().isGameOver();
  $('btn-stop').hidden = !running;
  const a = state.analysis;
  $('btn-play').disabled = !!running || !a || a.rootFen !== currentFen() || !(a.result?.best_move_uci || a.parsed?.best?.legal);
  $('btn-back').disabled = state.cursor === 0;
  $('btn-fwd').disabled = state.cursor >= state.fens.length - 1;
  $('fen-input').value = currentFen();
}

// ------------------------------------------------------------------ wiring
$('btn-analyze').onclick = () => analyze();
$('btn-stop').onclick = () => fetch('/api/stop', { method: 'POST' });
$('btn-play').onclick = playBest;
$('btn-back').onclick = () => goTo(state.cursor - 1);
$('btn-fwd').onclick = () => goTo(state.cursor + 1);
$('btn-flip').onclick = () => {
  state.orientation = state.orientation === 'white' ? 'black' : 'white';
  cg.set({ orientation: state.orientation });
  $('evalbar').classList.toggle('flipped', state.orientation === 'black');
  restoreView();
};
$('btn-new').onclick = () => { state.analysis = null; clearAnalysisPanel(); setPosition([START_FEN], [], [], 0); };
$('queen-side').onchange = (e) => {
  state.queenSide = e.target.value;
  if (state.queenSide === 'w' && state.orientation === 'white' || state.queenSide === 'b' && state.orientation === 'black') $('btn-flip').click();
  renderTurn(); showGame(); maybeQueenMoves();
};
$('fen-form').onsubmit = (e) => {
  e.preventDefault();
  const fen = $('fen-input').value.trim();
  try { new Chess(fen); } catch (err) { setStatus('error', 'Invalid FEN: ' + err.message); return; }
  setPosition([fen], [], [], 0);
};
$('preview-exit').onclick = unpin;
$('tog-raw').onchange = (e) => { state.rawView = e.target.checked; renderAnalysis(); };
$('tog-nums').onchange = (e) => { state.showNums = e.target.checked; restoreView(); };
$('step-first').onclick = () => { stopStepper(); stepTo(0); };
$('step-prev').onclick = () => { stopStepper(); stepTo(state.step.idx - 1); };
$('step-next').onclick = () => { stopStepper(); stepTo(state.step.idx + 1); };
$('step-play').onclick = playLine;
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'Escape') unpin();
  else if (e.key === 'ArrowLeft') state.pinned ? $('step-prev').click() : goTo(state.cursor - 1);
  else if (e.key === 'ArrowRight') state.pinned ? $('step-next').click() : goTo(state.cursor + 1);
});

function clearAnalysisPanel() {
  $('prose').innerHTML = '<p class="muted">Make a move or press <b>Analyze position</b>.</p>';
  for (const id of ['best-move', 'promising', 'critical', 'eval-number']) { $(id).textContent = '—'; $(id).classList.add('muted'); }
  $('best-source').textContent = ''; $('eval-text').textContent = '';
  setEvalBar(0); updateStepper(); renderLegendExample();
}

showGame(); renderMoveList(); renderTurn(); updateButtons(); updateStepper(); renderLegendExample();
loadInfo();

// Debug handle for the browser console.
window.queen = { state, parseAnalysis, tokenize, cg, renderAnalysis, setPosition, showGame, updateButtons };
