// plot.js — small canvas time-series and bar plots, themed by CSS variables.
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

function setupCanvas(canvas, height) {
  const w = canvas.clientWidth || 400, d = window.devicePixelRatio || 1;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(w * d);
  canvas.height = Math.round(height * d);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(d, 0, 0, d, 0, 0);
  return { ctx, w, h: height };
}

export class TimePlot {
  constructor(canvas, o) {
    this.c = canvas;
    this.o = Object.assign({ height: 140, window: 5, min: -1, max: 1, auto: false, minSpan: 1, maxSpan: Infinity, symmetric: true, hlines: [] }, o);
    this.data = [];
    this.lastW = 0;
    new ResizeObserver(() => { if (canvas.clientWidth !== this.lastW) { this.lastW = canvas.clientWidth; this.draw(this.tNow || 0); } }).observe(canvas);
  }
  push(t, vals) {
    this.data.push([t, ...vals]);
    const cut = t - this.o.window - 0.5;
    let i = 0;
    while (i < this.data.length && this.data[i][0] < cut) i++;
    if (i) this.data.splice(0, i);
  }
  clear() { this.data = []; }
  range() {
    const o = this.o;
    if (!o.auto) return [o.min, o.max];
    let m = 0;
    for (const row of this.data) for (let k = 1; k < row.length; k++) if (Number.isFinite(row[k])) m = Math.max(m, Math.abs(row[k]));
    const raw = Math.min(o.maxSpan, Math.max(o.minSpan, m * 1.15));
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const span = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((v) => v >= raw) || raw;
    return o.symmetric ? [-span, span] : [0, span];
  }
  draw(tNow) {
    this.tNow = tNow;
    const { ctx, w, h } = setupCanvas(this.c, this.o.height);
    const o = this.o;
    const L = 44, R = 10, T = 22, B = 8;
    const pw = w - L - R, ph = h - T - B;
    const [y0, y1] = this.range();
    const t0 = Math.max(0, tNow - o.window), t1 = Math.max(o.window, tNow);
    const X = (t) => L + ((t - t0) / (t1 - t0)) * pw;
    const Y = (v) => T + (1 - (clamp(v, y0, y1) - y0) / (y1 - y0)) * ph;
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillStyle = css("--ink-2");
    ctx.fillText(o.title || "", L, 14);
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right";
    for (const v of o.symmetric ? [y0, 0, y1] : [y0, (y0 + y1) / 2, y1]) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      const label = Math.abs(v) >= 10 || Number.isInteger(v) ? String(Math.round(v)) : v.toFixed(2);
      ctx.fillText(label, L - 6, y + 3.5);
    }
    ctx.textAlign = "left";
    for (const hl of o.hlines) {
      if (hl.v < y0 || hl.v > y1) continue;
      ctx.strokeStyle = css(hl.color || "--ink-3"); ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(L, Y(hl.v)); ctx.lineTo(L + pw, Y(hl.v)); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.lineWidth = 1.6; ctx.lineJoin = "round";
    o.series.forEach((s, k) => {
      ctx.strokeStyle = s.color.startsWith("--") ? css(s.color) : s.color;
      ctx.globalAlpha = s.alpha ?? 1;
      ctx.beginPath();
      let pen = false;
      for (const row of this.data) {
        const v = row[k + 1];
        if (row[0] < t0 || !Number.isFinite(v)) { pen = false; continue; }
        if (pen) ctx.lineTo(X(row[0]), Y(v)); else { ctx.moveTo(X(row[0]), Y(v)); pen = true; }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  }
}

// Horizontal bars for per-joint values in [lo, hi], with optional markers.
//   items: [{ label, value, lo, hi, warn }]
export function drawJointBars(canvas, items, o = {}) {
  const rowH = o.rowH || 18;
  const { ctx, w } = setupCanvas(canvas, items.length * rowH + (o.title ? 20 : 4));
  const L = o.labelW || 28, R = o.rightW ?? 52, top = o.title ? 20 : 2;
  ctx.font = "11px Inter, system-ui, sans-serif";
  if (o.title) { ctx.fillStyle = css("--ink-2"); ctx.fillText(o.title, L, 13); }
  items.forEach((it, i) => {
    const y = top + i * rowH;
    const pw = w - L - R;
    ctx.fillStyle = css("--ink-3");
    ctx.fillText(it.label, 2, y + rowH / 2 + 4);
    ctx.fillStyle = css("--track");
    ctx.fillRect(L, y + 4, pw, rowH - 8);
    const f = clamp((it.value - it.lo) / (it.hi - it.lo), 0, 1);
    if (o.mode === "fill") {
      ctx.fillStyle = it.warn ? css("--warn") : css(it.color || "--accent");
      ctx.fillRect(L, y + 4, pw * f, rowH - 8);
      if (o.limitAt != null) {
        const xl = L + pw * ((o.limitAt - it.lo) / (it.hi - it.lo));
        ctx.fillStyle = css("--warn"); ctx.fillRect(xl - 1, y + 2, 2, rowH - 4);
      }
    } else {
      // marker for position within joint range
      ctx.fillStyle = it.warn ? css("--warn") : css(it.color || "--accent");
      const x = L + pw * f;
      ctx.fillRect(x - 2, y + 2, 4, rowH - 4);
    }
    if (it.text) {
      ctx.fillStyle = it.warn ? css("--warn") : css("--ink-2");
      ctx.textAlign = "right";
      ctx.fillText(it.text, w - 2, y + rowH / 2 + 4);
      ctx.textAlign = "left";
    }
  });
}

// A heatmap of an n×n matrix (row-major), diverging colors around 0.
export function drawHeatmap(canvas, M, n, o = {}) {
  const size = Math.min(canvas.clientWidth || 300, o.maxSize || 320);
  const { ctx } = setupCanvas(canvas, size + 22);
  const L = 22, T = 18, cell = (size - L) / n;
  const vmax = o.vmax || Math.max(...M.map(Math.abs));
  const pos = css("--heat-pos"), neg = css("--heat-neg"), bg = css("--grid");
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillStyle = css("--ink-2");
  if (o.title) ctx.fillText(o.title, L, 12);
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = css("--ink-3");
    ctx.textAlign = "right"; ctx.fillText(String(i + 1), L - 6, T + (i + 0.5) * cell + 4);
    ctx.textAlign = "center"; ctx.fillText(String(i + 1), L + (i + 0.5) * cell, T + n * cell + 13);
    for (let j = 0; j < n; j++) {
      const v = M[i * n + j];
      const a = Math.min(1, Math.sqrt(Math.abs(v) / vmax));
      ctx.fillStyle = bg; ctx.fillRect(L + j * cell, T + i * cell, cell - 1, cell - 1);
      ctx.globalAlpha = a; ctx.fillStyle = v >= 0 ? pos : neg;
      ctx.fillRect(L + j * cell, T + i * cell, cell - 1, cell - 1);
      ctx.globalAlpha = 1;
      if (cell > 30) {
        ctx.fillStyle = a > 0.55 ? "#fff" : css("--ink-2");
        ctx.fillText(Math.abs(v) >= 0.995 ? v.toFixed(1) : v.toFixed(2).replace(/^(-?)0\./, "$1."), L + (j + 0.5) * cell, T + (i + 0.5) * cell + 4);
      }
    }
  }
  ctx.textAlign = "left";
}
