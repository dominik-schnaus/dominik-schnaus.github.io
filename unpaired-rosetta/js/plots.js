// Small hand-written SVG charts: the CKA scatter beside the cloud, the domain scatter, the few-pair sparkline.

const NS = "http://www.w3.org/2000/svg";
const el = (name, attributes = {}, parent) => {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attributes)) node.setAttribute(k, v);
  if (parent) parent.append(node);
  return node;
};

function frame(host, margin) {
  host.innerHTML = "";
  const style = getComputedStyle(host);  // the content box, so that nothing sticks out of the padded card
  const width = (host.clientWidth || 320) - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const height = (host.clientHeight || 160) - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height }, host);
  return { svg, width, height, x0: margin.left, x1: width - margin.right, y0: height - margin.bottom, y1: margin.top };
}

function axes(f, xDomain, yDomain, { xTicks, yTicks, xLabel, yLabel, fontSize = 10 }) {
  const sx = (v) => f.x0 + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * (f.x1 - f.x0);
  const sy = (v) => f.y0 - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * (f.y0 - f.y1);
  const g = el("g", { "font-size": fontSize, stroke: "none" }, f.svg);
  el("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, stroke: "var(--line)" }, g);
  el("line", { x1: f.x0, x2: f.x0, y1: f.y0, y2: f.y1, stroke: "var(--line)" }, g);
  for (const t of xTicks) {
    el("line", { x1: sx(t), x2: sx(t), y1: f.y0, y2: f.y0 + 3, stroke: "var(--ink-3)" }, g);
    el("text", { x: sx(t), y: f.y0 + fontSize + 4, "text-anchor": "middle" }, g).textContent = t;
  }
  for (const t of yTicks) {
    el("line", { x1: f.x0 - 3, x2: f.x0, y1: sy(t), y2: sy(t), stroke: "var(--ink-3)" }, g);
    el("text", { x: f.x0 - 5, y: sy(t) + fontSize * 0.35, "text-anchor": "end" }, g).textContent = t;
  }
  if (xLabel) el("text", { x: (f.x0 + f.x1) / 2, y: f.height - 2, "text-anchor": "middle", "font-weight": 500 }, g).textContent = xLabel;
  if (yLabel) {
    const t = el("text", { x: 0, y: 0, "text-anchor": "middle", "font-weight": 500, transform: `translate(${fontSize + 1},${(f.y0 + f.y1) / 2}) rotate(-90)` }, g);
    t.textContent = yLabel;
  }
  return { sx, sy };
}

function band(f, s, fit, key, { opacity = 0.1 } = {}) {
  const top = fit.map((r) => `${s.sx(r[key])},${s.sy(r.high)}`);
  const bottom = fit.slice().reverse().map((r) => `${s.sx(r[key])},${s.sy(r.low)}`);
  const clip = el("clipPath", { id: `clip${Math.random().toString(36).slice(2, 8)}` }, f.svg);
  el("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1 }, clip);
  const g = el("g", { "clip-path": `url(#${clip.id})` }, f.svg);
  el("polygon", { points: [...top, ...bottom].join(" "), fill: "var(--ink)", opacity }, g);
  const line = el("polyline", { points: fit.map((r) => `${s.sx(r[key])},${s.sy(r.fit)}`).join(" "), fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.4 }, g);
  return { group: g, line };
}

export const DATASET_COLORS = {
  coco_train2014: "var(--d-blue)", StanfordParagraphCaptioning: "var(--d-orange)",
  DenselyCaptionedImages: "var(--d-green)", DOCCIDataset: "var(--d-purple)",
};

// The 84 vision-language combinations: FOSCTTM against CKA.
export function ckaScatter(host, points, fit, { onPick } = {}) {
  const f = frame(host, { left: 40, right: 10, top: 10, bottom: 34 });
  const narrow = f.width < 300;
  const s = axes(f, [0.1, 0.75], [0, 0.6], { xTicks: narrow ? [0.2, 0.4, 0.6] : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], yTicks: [0, 0.2, 0.4, 0.6], xLabel: narrow ? "CKA" : "CKA (shared geometry)", yLabel: "FOSCTTM", fontSize: 10 });
  const names = { coco_train2014: "MS COCO", StanfordParagraphCaptioning: "SPC", DenselyCaptionedImages: "DCI", DOCCIDataset: "DOCCI" };
  Object.entries(names).forEach(([key, label], k) => {
    const step = narrow ? 52 : 72;
    const x = f.x1 - 2 * step - 6 + (k % 2) * step, y = f.y1 + 6 + Math.floor(k / 2) * 14;
    el("circle", { cx: x, cy: y - 3.5, r: 3.5, fill: DATASET_COLORS[key] }, f.svg);
    el("text", { x: x + 7, y, "font-size": 10.5 }, f.svg).textContent = label;
  });
  const trend = band(f, s, fit, "cka");
  trend.group.style.transition = "opacity .6s";
  trend.group.style.opacity = 0;
  const dots = points.map((p, i) => {
    const c = el("circle", { cx: s.sx(p.cka), cy: s.sy(p.foscttm), r: 4, fill: DATASET_COLORS[p.dataset], opacity: 0.55, style: "cursor:pointer" }, f.svg);
    c.addEventListener("click", () => onPick?.(i));
    return c;
  });
  const ring = el("circle", { r: 8, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5 }, f.svg);
  return {
    highlight(i) {
      dots.forEach((d, j) => d.setAttribute("opacity", j === i ? 1 : 0.45));
      ring.setAttribute("cx", s.sx(points[i].cka));
      ring.setAttribute("cy", s.sy(points[i].foscttm));
    },
    showTrend(on) { trend.group.style.opacity = on ? 1 : 0; },
    dotCenter(i) {
      const r = dots[i].getBoundingClientRect();
      return [r.left + r.width / 2, r.top + r.height / 2];
    },
  };
}

// Few-pair curve: ours against the best of seven baselines. Both axes are logarithmic (pairs + 1 on x), so that
// the gap at a handful of pairs is visible.
export function sparkline(host, curve) {
  const f = frame(host, { left: 44, right: 12, top: 12, bottom: 36 });
  const lx = (k) => Math.log10(k + 1), ly = (v) => Math.log10(Math.max(v, 0.002));
  const s = axes(f, [0, lx(1000)], [ly(0.002), ly(0.7)], { xTicks: [], yTicks: [], xLabel: "number of known pairs", yLabel: "FOSCTTM", fontSize: 10 });
  const narrow = f.width < 300;
  for (const k of narrow ? [0, 10, 1000] : [0, 1, 10, 100, 1000]) {
    el("line", { x1: s.sx(lx(k)), x2: s.sx(lx(k)), y1: f.y0, y2: f.y0 + 3, stroke: "var(--ink-3)" }, f.svg);
    el("text", { x: s.sx(lx(k)), y: f.y0 + 14, "font-size": 10, "text-anchor": "middle" }, f.svg).textContent = k;
  }
  for (const v of [0.003, 0.01, 0.03, 0.1, 0.3]) {
    el("line", { x1: f.x0, x2: f.x1, y1: s.sy(ly(v)), y2: s.sy(ly(v)), stroke: "var(--line)" }, f.svg);
    el("text", { x: f.x0 - 5, y: s.sy(ly(v)) + 3.5, "font-size": 10, "text-anchor": "end" }, f.svg).textContent = v;
  }
  el("line", { x1: f.x0, x2: f.x1, y1: s.sy(ly(0.5)), y2: s.sy(ly(0.5)), stroke: "var(--ink-3)", "stroke-dasharray": "3 4" }, f.svg);
  el("text", { x: f.x1 - 2, y: s.sy(ly(0.5)) - 4, "font-size": 10, "text-anchor": "end" }, f.svg).textContent = "chance";
  const path = (values, color, width, dash) => el("polyline", {
    points: curve.num_pairs.map((k, i) => `${s.sx(lx(k))},${s.sy(ly(values[i]))}`).join(" "),
    fill: "none", stroke: color, "stroke-width": width, "stroke-linejoin": "round", "stroke-dasharray": dash || "none",
  }, f.svg);
  path(curve.best_baseline, "var(--ink-3)", 1.8, "5 3");
  path(curve.ours, "var(--ochre)", 2.4);
  const legend = (y, color, dash, text) => {
    const x0 = f.x1 - (narrow ? 88 : 150);
    el("line", { x1: x0, x2: x0 + 18, y1: y, y2: y, stroke: color, "stroke-width": 2.2, "stroke-dasharray": dash || "none" }, f.svg);
    el("text", { x: x0 + 23, y: y + 3.5, "font-size": 10 }, f.svg).textContent = text;
  };
  const top = s.sy(ly(0.5)) + 16;  // just under the chance line, at the right: both curves are low there
  legend(top, "var(--ochre)", null, "ours");
  legend(top + 14, "var(--ink-3)", "5 3", narrow ? "best baseline" : "best of seven baselines");
  const marker = el("circle", { r: 5, fill: "var(--ochre)", stroke: "var(--paper-3)", "stroke-width": 1.5 }, f.svg);
  return { mark(i) { marker.setAttribute("cx", s.sx(lx(curve.num_pairs[i]))); marker.setAttribute("cy", s.sy(ly(curve.ours[i]))); } };
}

const DOMAIN_SHAPES = { language: "circle", biology: "square", medicine: "triangle", neuroscience: "diamond", chemistry: "pentagon", "materials science": "hexagon", astronomy: "star" };
const DOMAIN_COLORS = { language: "var(--d-blue)", biology: "var(--d-green)", medicine: "var(--d-red)", neuroscience: "var(--d-yellow)", chemistry: "var(--d-purple)", "materials science": "var(--d-sand)", astronomy: "var(--d-pink)" };

function marker(shape, x, y, r) {
  const poly = (n, rot = -Math.PI / 2, inner = null) => {
    const pts = [];
    const count = inner ? n * 2 : n;
    for (let k = 0; k < count; k++) {
      const radius = inner && k % 2 ? inner : r;
      const a = rot + (k / count) * Math.PI * 2;
      pts.push(`${(x + radius * Math.cos(a)).toFixed(1)},${(y + radius * Math.sin(a)).toFixed(1)}`);
    }
    return el("polygon", { points: pts.join(" ") });
  };
  switch (shape) {
    case "square": return el("rect", { x: x - r * 0.85, y: y - r * 0.85, width: r * 1.7, height: r * 1.7, rx: 1.5 });
    case "triangle": return poly(3);
    case "diamond": return poly(4);
    case "pentagon": return poly(5);
    case "hexagon": return poly(6);
    case "star": return poly(5, -Math.PI / 2, r * 0.45);
    default: return el("circle", { cx: x, cy: y, r });
  }
}

// Fourteen modality pairs from seven fields.
export function domainScatter(host, domains, fit, vl) {
  const small = host.clientWidth < 600;
  const f = frame(host, { left: small ? 44 : 60, right: small ? 12 : 150, top: 14, bottom: 44 });
  const s = axes(f, [0, 1], [-0.02, 0.62], { xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], xLabel: "CKA (shared geometry)", yLabel: "FOSCTTM (lower is better)", fontSize: small ? 10 : 12 });
  band(f, s, fit, "cka_unbiased");
  el("line", { x1: f.x0, x2: f.x1, y1: s.sy(0.5), y2: s.sy(0.5), stroke: "var(--ink-3)", "stroke-dasharray": "3 4" }, f.svg);
  // on small screens the legend takes the empty top right corner, so the chance label moves to the middle
  el("text", { x: small ? s.sx(0.5) : f.x1 - 4, y: s.sy(0.5) + (small ? 13 : -5), "text-anchor": small ? "middle" : "end", "font-size": 11 }, f.svg).textContent = "chance";
  // The vision-language combinations of the section before, as faint context.
  for (const p of vl) el("circle", { cx: s.sx(p.cka), cy: s.sy(p.foscttm), r: 2.2, fill: "var(--ink-3)", opacity: 0.28 }, f.svg);
  const tip = document.createElement("div");
  tip.className = "tip"; tip.hidden = true; document.body.append(tip);
  for (const d of domains) {
    const node = marker(DOMAIN_SHAPES[d.domain], s.sx(d.cka_unbiased), s.sy(d.foscttm), small ? 5.5 : 7);
    node.setAttribute("fill", DOMAIN_COLORS[d.domain]);
    node.setAttribute("stroke", "var(--paper-3)");
    node.setAttribute("stroke-width", 1.5);
    node.style.cursor = "help";
    f.svg.append(node);
    node.addEventListener("pointerenter", (e) => {
      tip.innerHTML = `<b>${d.label}</b><br><span style="color:var(--ink-3)">${d.domain} · CKA ${d.cka_unbiased.toFixed(2)} · FOSCTTM ${d.foscttm < 0.001 ? d.foscttm.toExponential(1) : d.foscttm.toFixed(3)}</span>`;
      tip.hidden = false;
      tip.style.left = `${e.clientX + 12}px`; tip.style.top = `${e.clientY + 12}px`;
    });
    node.addEventListener("pointerleave", () => { tip.hidden = true; });
  }
  // Labels: greedy placement around each marker, skipping any that would collide (the tooltip still names it).
  if (!small) {
    const boxes = domains.map((d) => { const x = s.sx(d.cka_unbiased), y = s.sy(d.foscttm); return [x - 8, y - 8, x + 8, y + 8]; });
    const hit = (b) => b[0] < f.x0 || b[2] > f.x1 + 140 || b[1] < f.y1 - 6 || b[3] > f.y0 || boxes.some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]);
    const order = domains.map((d, i) => i).sort((i, j) => domains[i].cka_unbiased - domains[j].cka_unbiased);
    for (const i of order) {
      const d = domains[i], x = s.sx(d.cka_unbiased), y = s.sy(d.foscttm), w = d.label.length * 6.1, h = 13;
      const candidates = [[x + 10, y + 4, "start"], [x - 10, y + 4, "end"], [x, y - 11, "middle"], [x, y + 18, "middle"], [x + 8, y - 9, "start"], [x + 8, y + 16, "start"], [x - 8, y - 9, "end"], [x - 8, y + 16, "end"]];
      for (const [tx, ty, anchor] of candidates) {
        const left = anchor === "start" ? tx : anchor === "end" ? tx - w : tx - w / 2;
        const box = [left, ty - h + 3, left + w, ty + 3];
        if (hit(box)) continue;
        boxes.push(box);
        el("text", { x: tx, y: ty, "font-size": 11, "text-anchor": anchor }, f.svg).textContent = d.label;
        break;
      }
    }
  }
  // Legend: the seven fields.
  const legend = el("g", { "font-size": 11.5 }, f.svg);
  const fields = [...new Set(domains.map((d) => d.domain))];
  fields.forEach((field, k) => {
    const col = Math.min(104, (f.x1 - f.x0) / 2.6);
    const x = small ? f.x1 - 2 * col + (k % 2) * col : f.x1 + 22, y = small ? f.y1 + 4 + Math.floor(k / 2) * 14 : f.y1 + 12 + k * 20;
    const m = marker(DOMAIN_SHAPES[field], x, y - 4, 5);
    m.setAttribute("fill", DOMAIN_COLORS[field]);
    legend.append(m);
    el("text", { x: x + 10, y }, legend).textContent = field;
  });
  if (!small) {
    const y = f.y1 + 12 + fields.length * 20 + 8;
    el("circle", { cx: f.x1 + 22, cy: y - 4, r: 2.4, fill: "var(--ink-3)", opacity: 0.4 }, legend);
    el("text", { x: f.x1 + 32, y }, legend).textContent = "image ↔ text (IV)";
  }
}

// Results: how the per-image fractions are spread. Their mean is the FOSCTTM of the shown pairs.
export function histogram(host) {
  const f = frame(host, { left: 40, right: 12, top: 12, bottom: 34 });
  const bins = 40;
  const sx = (v) => f.x0 + v * (f.x1 - f.x0);
  const g = el("g", {}, f.svg);
  el("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, stroke: "var(--line)" }, f.svg);
  const narrow = f.width < 330;
  for (const t of narrow ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]) {
    el("text", { x: sx(t), y: f.y0 + 14, "font-size": 10, "text-anchor": "middle" }, f.svg).textContent = t;
  }
  el("text", { x: (f.x0 + f.x1) / 2, y: f.height - 3, "font-size": 10.5, "text-anchor": "middle", "font-weight": 500 }, f.svg).textContent = narrow ? "captions closer than the true one" : "fraction of captions closer than the true caption, per image";
  const yl = el("text", { x: 0, y: 0, "font-size": 10.5, "text-anchor": "middle", "font-weight": 500, transform: `translate(12,${(f.y0 + f.y1) / 2}) rotate(-90)` }, f.svg);
  yl.textContent = "images";
  el("line", { x1: sx(0.5), x2: sx(0.5), y1: f.y0, y2: f.y1, stroke: "var(--ink-3)", "stroke-dasharray": "3 4" }, f.svg);
  el("text", { x: sx(0.5) + 4, y: f.y1 + 10, "font-size": 10 }, f.svg).textContent = "chance";
  const meanLine = el("line", { y1: f.y0, y2: f.y1, stroke: "var(--ochre)", "stroke-width": 1.6 }, f.svg);
  const meanText = el("text", { y: f.y1 + 24, "font-size": 10.5, fill: "var(--ochre)" }, f.svg);
  const mark = el("path", { fill: "var(--ink)" }, f.svg);
  return {
    update(values, marked) {
      const counts = new Array(bins).fill(0);
      let sum = 0;
      for (const v of values) { counts[Math.min(bins - 1, Math.floor(v * bins))]++; sum += v; }
      const top = Math.max(...counts);
      const h = (c) => (c ? Math.max(1.5, Math.sqrt(c / top) * (f.y0 - f.y1)) : 0);  // square root, so the tail stays visible
      g.innerHTML = "";
      counts.forEach((c, b) => el("rect", { x: sx(b / bins) + 0.5, width: (f.x1 - f.x0) / bins - 1, y: f.y0 - h(c), height: h(c), fill: "var(--ink-2)", opacity: 0.55 }, g));
      const m = sum / values.length;
      meanLine.setAttribute("x1", sx(m)); meanLine.setAttribute("x2", sx(m));
      meanText.setAttribute("x", sx(m) + 5); meanText.textContent = `mean ${m < 0.1 ? m.toFixed(3) : m.toFixed(2)}${narrow ? "" : " (FOSCTTM)"}`;
      if (marked === null || marked === undefined) mark.setAttribute("d", "");
      else { const x = sx(marked); mark.setAttribute("d", `M${x},${f.y0 + 2} l5,8 h-10 z`); }
    },
  };
}

