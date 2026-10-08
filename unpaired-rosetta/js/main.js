// An atlas around one point cloud. The cloud fills the screen and can always be moved, zoomed and inspected;
// chapters (in a side panel on wide screens, a bottom sheet on phones) only change what it shows. Every scene maps
// its state (and its controls) to screen positions and colours of the same pool of points, and the stage draws them.
// Slots: 0..n-1 are images (circles), n..2n-1 captions (triangles).

import { Stage, SHAPE } from "./stage.js";
import { Sheet } from "./sheet.js";
import { loadManifest, loadMethod, loadValBlock, loadT2I, loadJSON, fetchOnce } from "./data.js";
import { carve, carveTitle, dialogs, reducedMotion } from "./ui.js";
import { ckaScatter, sparkline, domainScatter, histogram } from "./plots.js";

// ------------------------------------------------------------------------------------------------ helpers
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const smooth = (a, b, x) => ease(clamp((x - a) / (b - a)));
const lerp = (a, b, t) => a + (b - a) * t;
const mix3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const $ = (id) => document.getElementById(id);
// Phones, and tablets held upright, get the bottom sheet; everything wider gets the side panel.
const SHEET = "(max-width: 600px), (orientation: portrait) and (max-width: 1000px)";
const sheetQuery = matchMedia(SHEET);
const isMobile = () => sheetQuery.matches;
const fmt = (v) => (v < 0.001 ? v.toExponential(1) : v < 0.1 ? v.toFixed(3) : v.toFixed(2));
const pct = (v) => (v < 0.001 ? "<0.1" : v < 0.1 ? (v * 100).toFixed(1) : (v * 100).toFixed(0));
const rgb = (c, a = 1) => `rgba(${c.map((v) => Math.round(v * 255)).join(",")},${a})`;
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
function mulberry(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const mean = (a) => { let x = 0, y = 0; for (let i = 0; i < a.length; i += 2) { x += a[i]; y += a[i + 1]; } return [(2 * x) / a.length, (2 * y) / a.length]; };

function cssColor(name) {
  const probe = document.createElement("i");
  probe.style.color = `var(${name})`;
  document.body.append(probe);
  const [r, g, b] = getComputedStyle(probe).color.match(/[\d.]+/g).map(Number);
  probe.remove();
  return [r / 255, g / 255, b / 255];
}
let CLUSTER = [];
// COCO supercategories as in Fig. 6(c): animals, vehicles, food, sports, furniture, people; 6 = other or mixed.
const SUPER_NAMES = ["animals", "vehicles", "food", "sports", "furniture", "people"];
let C = {}, SUPER = [];
const readColors = () => {
  C = { neutral: cssColor("--ink-2"), ink: cssColor("--ink"), ink3: cssColor("--ink-3"), stoneInk: cssColor("--stone-ink"), paper: cssColor("--paper-3") };
  // data colours come from the theme: brighter on the dark stone, deeper on the light one
  const d = (name) => cssColor(`--d-${name}`);
  SUPER = [d("green"), d("pink"), d("orange"), d("blue"), d("purple")].concat([C.neutral, C.ink3]);
  CLUSTER = [d("orange"), d("blue"), d("green"), d("pink"), d("yellow"), d("sand"), d("red"), d("purple")];
};

// A frame: everything the stage needs for every point.
const makeFrame = (n) => ({ pos: new Float32Array(2 * n), col: new Float32Array(4 * n), size: new Float32Array(n), shape: new Float32Array(n) });
function blend(out, a, b, t) {
  for (let i = 0; i < out.pos.length; i++) out.pos[i] = lerp(a.pos[i], b.pos[i], t);
  for (let i = 0; i < out.col.length; i++) out.col[i] = lerp(a.col[i], b.col[i], t);
  for (let i = 0; i < out.size.length; i++) { out.size[i] = lerp(a.size[i], b.size[i], t); out.shape[i] = t < 0.5 ? a.shape[i] : b.shape[i]; }
}
const copy = (out, a) => { out.pos.set(a.pos); out.col.set(a.col); out.size.set(a.size); out.shape.set(a.shape); };
const setColor = (fr, i, c, alpha) => { fr.col[4 * i] = c[0]; fr.col[4 * i + 1] = c[1]; fr.col[4 * i + 2] = c[2]; fr.col[4 * i + 3] = alpha; };
const SIZE = { image: 6.4, text: 8.4 };
// Larger plots (desktops) get larger symbols; set once per frame from the plot's size.
let pointScale = 1;
const marker = (fr, i, scale = 1) => { const image = i < n; fr.size[i] = (image ? SIZE.image : SIZE.text) * scale * pointScale; fr.shape[i] = image ? SHAPE.image : SHAPE.text; };

// ------------------------------------------------------------------------------------------------ state
let M, D, T2I, stage, sheet, N, n;
let captionsTrain = null, captionsVal = null, captionsT2I = null, altTrain = null;
let superTrain, superVal, superT2I;  // Uint8Array labels per slot / validation index / t2i row
const valBlocks = new Map();
const hover = { slot: -1 };
let activeScene = null;
let animatingUntil = 0;
const bump = (ms = 1200) => { animatingUntil = Math.max(animatingUntil, performance.now() + ms); requestFrame(); };

// ------------------------------------------------------------------------------------------------ layout
// The plot is whatever the panel leaves free: right of the side panel, or above the sheet at its current height.
// A row above the plot holds the setting's name, the legend and the FOSCTTM.
// The row is as tall as the full legend at the plot's width (measured once per width, so that the cloud does not
// jump when the legend gains its colours), plus the setting's name on wide screens (phones show it in the sheet).
const STAGE_LINE = 28;
let legendMeasure = { key: "", h: 0 };
function labelRow(width) {
  const key = `${Math.round(width)}|${isMobile()}|${innerHeight < 560}`;
  if (legendMeasure.key !== key) {
    const m = $("legend-measure");
    if (!m.childElementCount) m.innerHTML = legendHTML("known");
    m.style.maxWidth = `${width}px`;
    legendMeasure = { key, h: m.offsetHeight };
  }
  return (isMobile() ? 0 : STAGE_LINE) + legendMeasure.h + 12;
}
function plotRect() {
  const W = innerWidth, H = innerHeight, top = $("topbar")?.offsetHeight || 48;
  let x, w, bottom;
  if (isMobile()) {
    x = 12; w = W - 24;
    bottom = H - Math.min(sheet?.height() || 0, H * 0.62) - 12;
  } else {
    x = Math.max(24, $("panel").getBoundingClientRect().right + 28);
    w = Math.max(120, W - x - 28);
    bottom = H - 24;
  }
  const y = top + labelRow(w);
  return { x, y, w, h: Math.max(isMobile() ? 90 : 120, bottom - y) };
}

// Data units -> screen. One fixed box for all layouts (so moving between chapters never rescales the cloud),
// fitted into the plot, then the reader's camera: zoom around the plot's centre and a shift.
let BOX;
const cam = { z: 1, x: 0, y: 0 };
const camGoal = { z: 1, x: 0, y: 0, until: 0 };
function viewFor(rect, box = BOX) {
  const s = Math.min(rect.w / (box.x1 - box.x0), rect.h / (box.y1 - box.y0)) * 0.96;
  const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
  const ox = rect.x + rect.w / 2, oy = rect.y + rect.h / 2, z = cam.z, dx = cam.x, dy = cam.y;
  return { s: s * z, map: (x, y) => [ox + (x - cx) * s * z + dx, oy - (y - cy) * s * z + dy] };
}
function boxOf(...arrays) {
  const box = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
  for (const a of arrays) for (let i = 0; i < a.length; i += 2) {
    box.x0 = Math.min(box.x0, a[i]); box.x1 = Math.max(box.x1, a[i]);
    box.y0 = Math.min(box.y0, a[i + 1]); box.y1 = Math.max(box.y1, a[i + 1]);
  }
  return box;
}
const camMoved = () => Math.abs(cam.z - 1) > 0.01 || Math.abs(cam.x) > 2 || Math.abs(cam.y) > 2;
function resetCamera(animate = true) {
  if (!animate || reducedMotion) { cam.z = 1; cam.x = 0; cam.y = 0; camGoal.until = 0; requestFrame(); return; }
  Object.assign(camGoal, { z: 1, x: 0, y: 0, until: performance.now() + 600, from: { ...cam }, start: performance.now() });
  requestFrame();
}
function stepCamera(now) {
  if (!camGoal.until) return false;
  const t = ease(clamp((now - camGoal.start) / 600));
  cam.z = lerp(camGoal.from.z, camGoal.z, t); cam.x = lerp(camGoal.from.x, camGoal.x, t); cam.y = lerp(camGoal.from.y, camGoal.y, t);
  if (t >= 1) camGoal.until = 0;
  return t < 1;
}
// Zoom by a factor around a screen point, keeping the point under the finger or the pointer in place.
function zoomAt(factor, sx, sy) {
  const r = plotRect(), ox = r.x + r.w / 2, oy = r.y + r.h / 2;
  const z = clamp(cam.z * factor, 0.6, 10), k = z / cam.z;
  cam.x = sx - ox - (sx - ox - cam.x) * k;
  cam.y = sy - oy - (sy - oy - cam.y) * k;
  cam.z = z; camGoal.until = 0;
}

const imageUrl = (kind, row) => `data/img/${kind}${row}.webp`;
async function block(url) {
  if (!valBlocks.has(url)) valBlocks.set(url, await loadValBlock(url, M));
  return valBlocks.get(url);
}

// ------------------------------------------------------------------------------------------------ chapters
// The title, then seven chapters. Each chapter has a scene (text to image is the few-pair scene's second step).
const CHAPTERS = ["title", "overview", "method", "results", "pairs", "t2i", "geometry", "domains", "benchmarks"];
const SCENE_OF = { title: "hero", overview: "overview", method: "method", results: "results", pairs: "pairs", t2i: "pairs", geometry: "geometry", domains: "domains", benchmarks: "benchmarks" };
let chapter = "title";
const SNAP = { frame: null, t: 1, last: 0 };  // the frame on screen when the chapter changed, and the move from it

function setChapter(id, step = 0, { push = true, instant = false, morph = true } = {}) {
  if (!CHAPTERS.includes(id)) id = "title";
  const from = chapter;
  if (id === from && (id !== "method" || scenes.method.manual === step) && !instant) return;
  autoplay.on = false;
  if (id === from && id === "method") { setStep(step); return; }
  // leaving the title plays the stone's own flight into the cloud; the chapter takes over when it lands
  if (from === "title" && id !== "title" && !instant && !reducedMotion) {
    scenes.method.reset();
    document.body.dataset.chapter = id;
    showArticle(id, step);
    if (isMobile()) sheet.set("half");
    scenes.hero.leave(() => setChapter(id, step, { push, instant: true }));
    return;
  }
  const sceneFrom = SCENE_OF[from], sceneTo = SCENE_OF[id];
  // morph from whatever was on screen (the title's flight already lands on the method's first step)
  if (sceneTo !== sceneFrom && morph && !(from === "title" && (sceneTo === "method" || sceneTo === "overview") && instant)) {
    copy(SNAP.frame, out.frame); SNAP.t = 0; SNAP.last = performance.now();
  }
  if (id === "title") scenes.hero.arrive(from !== "title");
  chapter = id;
  const scene = scenes[sceneTo];
  if (scene) {
    scene.manual = id === "t2i" ? 1 : id === "method" ? step : 0;
    if (sceneTo !== sceneFrom) scene.onEnter?.(scene.manual);
  }
  document.body.dataset.chapter = id;
  if (from !== id) $("panel-body").scrollTop = 0;
  showArticle(id, step);
  if (isMobile() && id !== "title") sheet.set(from === "title" && sheet.state === "peek" ? "half" : sheet.state);  // the peek height depends on the chapter
  if (push) history.pushState(null, "", id === "title" ? location.pathname : `#${id}${id === "method" && step ? `-${step}` : ""}`);
  hover.slot = -1; clearCard();
  if (sceneTo !== sceneFrom) resetCamera();
  sizeFigure();
  bump(700);  // frames through the panel's slide, which moves the plot
}
function showArticle(id, step) {
  for (const a of document.querySelectorAll("article[data-chapter]")) a.classList.toggle("active", a.dataset.chapter === id);
  for (const b of document.querySelectorAll("#chapters [data-go]")) b.toggleAttribute("aria-current", b.dataset.go === id);
  const title = document.querySelector(`article[data-chapter="${id}"] .carve`);
  if (title) carveTitle(title);
  // the chip of the current chapter stays in view when the chips scroll sideways (phones)
  const chips = $("chapters"), chip = chips.querySelector(`[data-go="${id}"]`);
  if (chip && chips.scrollWidth > chips.clientWidth) chips.scrollTo({ left: chip.offsetLeft - (chips.clientWidth - chip.offsetWidth) / 2, behavior: reducedMotion ? "auto" : "smooth" });
  if (id === "method") showStep(step);
}
function showStep(k) {
  for (const t of document.querySelectorAll('article[data-chapter="method"] .step-text')) t.classList.toggle("active", +t.dataset.step === k);
  for (const b of document.querySelectorAll('article[data-chapter="method"] .seg [data-step]')) b.setAttribute("aria-selected", +b.dataset.step === k);
}
function setStep(k, { fromPlay = false } = {}) {
  if (!fromPlay) autoplay.on = false;
  if (chapter !== "method") { setChapter("method", k); return; }
  scenes.method.manual = k;
  showStep(k);
  history.replaceState(null, "", `#method${k ? `-${k}` : ""}`);
  hover.slot = -1; clearCard();
  requestFrame();
}
// Play runs the method by itself: clustering, read-out and refinement, each once the one before has settled.
const autoplay = { on: false, since: 0 };
function playMethod() {
  if (chapter !== "method") { setChapter("method", 0); return; }
  copy(SNAP.frame, out.frame); SNAP.t = 0; SNAP.last = performance.now();  // rewind as a morph, not a jump
  scenes.method.reset();
  setStep(1);
  autoplay.on = true; autoplay.since = performance.now();
  bump(400);
}
function stepAutoplay(now) {
  if (!autoplay.on || chapter !== "method") return false;
  if (SNAP.t < 1) { autoplay.since = now; return true; }
  const m = scenes.method, k = m.manual;
  if (!m.done(k)) { autoplay.since = now; return true; }
  if (k >= 3) { autoplay.on = false; return false; }
  if (now - autoplay.since > (k === 1 ? 1500 : 1100)) setStep(k + 1, { fromPlay: true });
  return true;
}
// Replay restarts the animation of the current chapter.
function replay() {
  const scene = scenes[SCENE_OF[chapter]];
  if (scene?.replay) scene.replay(scene.manual);
  bump(300);
}

// ------------------------------------------------------------------------------------------------ colours
// neutral -> clusters (initialization and read-out) -> supercategories (from the refinement on)
function superOf(slot, context) {
  const k = slot % n;
  if (context === "train") return superTrain[slot];
  if (context === "t2i") return superT2I[k];
  return superVal[slot < n ? D.toValImage[k] : D.toValText[k]];
}
const superColor = (label) => SUPER[label];
const superAlpha = (label, alpha) => (label === 6 ? alpha * 0.45 : alpha);

// ------------------------------------------------------------------------------------------------ hover cards
// One card format everywhere: images on the left of the point, captions on the right, as in the paper's teaser.
function neighbourCard(image, item, neighbours, { imageOf, captionOf, truth, closer }) {
  const match = (j) => truth !== undefined && j === truth.index;
  if (image) {
    return {
      kind: "image",
      left: [{ img: imageOf(item), alt: captionOf(item, true), label: "image" }],
      right: neighbours.map((j, r) => ({ text: captionOf(j), label: `nearest caption ${r + 1}`, match: match(j) })),
      truth: truth?.slot, closer, closerNoun: "captions",
    };
  }
  return {
    kind: "caption",
    left: neighbours.map((j, r) => ({ img: imageOf(j), alt: captionOf(j, true), small: true, label: `nearest image ${r + 1}`, match: match(j) })),
    right: [{ text: captionOf(item), label: "caption", main: true }],
    truth: truth?.slot, closer, closerNoun: "images",
  };
}

// ------------------------------------------------------------------------------------------------ scenes
const scenes = {};

// The title: images and caption words on the stone. They drift in, settle into two matching constellations and
// link up; the pointer (or a finger) pushes them away, and they spring back.
function hero() {
  const section = $("hero");
  const wrap = $("stone-wrap"), stone = section.querySelector(".stone");
  const text = section.querySelector(".hero-text");
  const random = mulberry(7);
  const seeds = Array.from({ length: N }, () => [random(), random(), random() * 6.28, 0.5 + random()]);
  // the 2 to 98 % box of each cloud, so that a few outliers do not shrink the constellation
  const quantileBox = (a) => {
    const xs = [], ys = [];
    for (let i = 0; i < a.length; i += 2) { xs.push(a[i]); ys.push(a[i + 1]); }
    xs.sort((u, v) => u - v); ys.sort((u, v) => u - v);
    const q = (arr, t) => arr[Math.floor(t * (arr.length - 1))];
    return { x0: q(xs, 0.02), x1: q(xs, 0.98), y0: q(ys, 0.02), y1: q(ys, 0.98) };
  };
  const boxI = quantileBox(D.unalignedImage), boxT = quantileBox(D.unalignedText);
  const scratch = makeFrame(N), target = makeFrame(N);
  const heroImage = new Map(M.hero.image_slots.map((slot, k) => [slot, k]));
  const heroText = new Map(M.hero.text_slots.map((slot, k) => [slot, k]));
  const atlas = new Image();
  atlas.src = "data/hero_atlas.webp";
  atlas.onload = () => requestFrame();
  let centroids = null;
  const state = { p: 0 };

  // pointer physics: each particle has a displacement with a velocity and a spring back to its place
  const disp = new Float32Array(2 * N), vel = new Float32Array(2 * N);
  const pointer = { x: -1e4, y: -1e4, at: 0 };
  let energy = 0, lastPhys = performance.now();
  const onMove = (e) => {
    if (chapter !== "title") return;
    pointer.x = e.clientX; pointer.y = e.clientY; pointer.at = performance.now();
    requestFrame();
  };
  addEventListener("pointermove", onMove, { passive: true });
  addEventListener("pointerdown", onMove, { passive: true });
  addEventListener("pointerleave", () => { pointer.x = pointer.y = -1e4; });
  // a tap or click on the stone scatters everything a little, like a knock on the stone
  wrap.addEventListener("pointerdown", (e) => {
    const r = mulberry(Math.floor(performance.now()));
    for (let i = 0; i < N; i++) {
      const dx = scratch.pos[2 * i] - e.clientX, dy = scratch.pos[2 * i + 1] - e.clientY, d = Math.hypot(dx, dy) + 1;
      const kick = 900 * Math.exp(-d / 160) + 60 * r();
      vel[2 * i] += (dx / d) * kick; vel[2 * i + 1] += (dy / d) * kick;
    }
    requestFrame();
  });
  function physics(fr, now) {
    const dt = Math.min(0.034, (now - lastPhys) / 1000); lastPhys = now;
    const R = isMobile() ? 56 : 72, active = now - pointer.at < 1200;
    let e = 0;
    for (let i = 0; i < N; i++) {
      const bx = fr.pos[2 * i], by = fr.pos[2 * i + 1];
      let fx = -16 * disp[2 * i] - 5.5 * vel[2 * i], fy = -16 * disp[2 * i + 1] - 5.5 * vel[2 * i + 1];
      if (active) {
        const px = bx + disp[2 * i] - pointer.x, py = by + disp[2 * i + 1] - pointer.y, d = Math.hypot(px, py);
        if (d < R && d > 0.01) { const f = 2600 * (1 - d / R) ** 2; fx += (px / d) * f; fy += (py / d) * f; }
      }
      vel[2 * i] += fx * dt; vel[2 * i + 1] += fy * dt;
      disp[2 * i] = clamp(disp[2 * i] + vel[2 * i] * dt, -140, 140); disp[2 * i + 1] = clamp(disp[2 * i + 1] + vel[2 * i + 1] * dt, -140, 140);
      fr.pos[2 * i] = bx + disp[2 * i]; fr.pos[2 * i + 1] = by + disp[2 * i + 1];
      e += Math.abs(disp[2 * i]) + Math.abs(disp[2 * i + 1]);
    }
    energy = e / N;
    return active || energy > 0.05;
  }

  // the title's progress is time: arriving plays drift, settle and the lines (0 to REST); leaving plays the flight
  const REST = 0.57;
  const tween = { p: 0, target: REST, last: performance.now(), done: null };
  const progress = (now) => {
    const dt = Math.min(64, now - tween.last); tween.last = now;
    // per second: the first arrival takes its time, the flight into the cloud and the way back are quicker
    const rate = tween.target > REST ? 1 / 1.3 : tween.p > REST ? 1 / 2.4 : 1 / 5.4;
    tween.p = reducedMotion ? tween.target : tween.p < tween.target ? Math.min(tween.target, tween.p + rate * dt / 1000) : Math.max(tween.target, tween.p - rate * dt / 1000);
    if (tween.done && tween.p >= tween.target) { const done = tween.done; tween.done = null; setTimeout(done, 0); }
    return tween.p;
  };
  function inStone(fr, now, p) {
    const r = wrap.getBoundingClientRect();
    // two registers, as the scripts on the real stone: caption words above, images below (clear of the broken top)
    const bands = [{ x: r.left + r.width * 0.13, w: r.width * 0.74, y: r.top + r.height * 0.61, h: r.height * 0.27 },
                   { x: r.left + r.width * 0.14, w: r.width * 0.72, y: r.top + r.height * 0.25, h: r.height * 0.26 }];
    const settle = smooth(0.06, 0.42, p);
    const wobble = (1 - settle) * 5 + 0.6;  // a little life is always left in the particles
    // each cloud fills its register, stretched sideways by at most 1.7 so that it still reads as a constellation
    const fit = (box, band) => {
      const sy = band.h / (box.y1 - box.y0), sx = Math.min(band.w / (box.x1 - box.x0), 1.7 * sy);
      return (x, y) => [band.x + band.w / 2 + (x - (box.x0 + box.x1) / 2) * sx, band.y + band.h / 2 - (y - (box.y0 + box.y1) / 2) * sy];
    };
    const fits = [fit(boxI, bands[0]), fit(boxT, bands[1])];
    for (let i = 0; i < N; i++) {
      const side = i < n ? 0 : 1, k = i % n, [u, v, phase, speed] = seeds[i], band = bands[side];
      const src = side ? D.unalignedText : D.unalignedImage;
      const [cx, cy] = fits[side](src[2 * k], src[2 * k + 1]);
      const dx = band.x + u * band.w, dy = band.y + v * band.h;
      const t = (now / 1000) * speed * 0.5;
      const bx = clamp(cx, band.x, band.x + band.w), by = clamp(cy, band.y, band.y + band.h);
      fr.pos[2 * i] = lerp(dx, bx, settle) + wobble * Math.sin(t + phase);
      fr.pos[2 * i + 1] = lerp(dy, by, settle) + wobble * Math.cos(t * 1.3 + phase);
      setColor(fr, i, C.neutral, 0);  // the stone shows sprites, not points
      marker(fr, i);
    }
    const live = physics(fr, now);
    const sums = new Float32Array(32), counts = new Float32Array(16);
    for (let i = 0; i < N; i++) {
      const side = i < n ? 0 : 1, k = i % n;
      const c = (side ? D.clusterText[k] : D.clusterImage[k]) + side * 8;
      sums[2 * c] += fr.pos[2 * i]; sums[2 * c + 1] += fr.pos[2 * i + 1]; counts[c]++;
    }
    centroids = Array.from({ length: 16 }, (_, c) => [sums[2 * c] / counts[c], sums[2 * c + 1] / counts[c], counts[c]]);
    return live;
  }
  return {
    id: "hero", section,
    arrive(fromBelow) { tween.target = REST; if (fromBelow) tween.p = 1; tween.last = performance.now(); section.style.visibility = ""; requestFrame(); },
    skip() { tween.p = tween.target = 1; section.style.visibility = "hidden"; section.classList.add("away"); },
    leave(done) { tween.target = 1; tween.done = done; tween.last = performance.now(); requestFrame(); },
    replay() { tween.p = 0; tween.target = REST; tween.last = performance.now(); requestFrame(); },
    compute(fr, now) {
      const p = progress(now);
      state.p = p;
      const live = inStone(scratch, now, p);
      const exit = smooth(0.6, 1, p);
      stone.style.opacity = 1 - smooth(0.62, 0.85, p);
      wrap.style.setProperty("--fade", 1 - smooth(0.5, 0.75, p));
      section.querySelector(".stone-labels").style.opacity = 0.55 * (1 - smooth(0.15, 0.4, p));
      text.style.opacity = 1 - smooth(0.55, 0.8, p);
      text.style.transform = `translateY(${-30 * smooth(0.55, 0.85, p)}px)`;
      section.classList.toggle("away", p > 0.6);
      section.style.visibility = p >= 0.999 ? "hidden" : "";
      if (exit > 0) {
        scenes.method.state(target, 0, now);
        blend(fr, scratch, target, exit);
        for (let i = 0; i < N; i++) fr.col[4 * i + 3] = target.col[4 * i + 3] * smooth(0.62, 0.85, p);
      } else copy(fr, scratch);
      this.frame = fr;
      hud(false); setStageLabel(""); legend(null);
      // the particles always breathe on the title, so frames keep coming while it is open
      return !reducedMotion && (p !== tween.target || live || chapter === "title");
    },
    overlay(ctx) {
      const { p } = state, fr = this.frame;
      // the sprites: real images on one half, the most frequent words of the captions on the other
      const sprite = 1 - smooth(0.62, 0.82, p);
      if (sprite > 0) {
        const cell = M.hero.cell, cols = M.hero.cols, size = isMobile() ? 9 : 13;
        if (atlas.complete && atlas.naturalWidth) {
          ctx.globalAlpha = sprite;
          for (const [slot, k] of heroImage) {
            const x = fr.pos[2 * slot], y = fr.pos[2 * slot + 1];
            ctx.drawImage(atlas, (k % cols) * cell, Math.floor(k / cols) * cell, cell, cell, x - size / 2, y - size / 2, size, size);
          }
          ctx.globalAlpha = 1;
        }
        ctx.font = `500 ${isMobile() ? 7.5 : 10}px "Work Sans", system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.fillStyle = rgb(C.stoneInk, 0.9 * sprite);
        for (const [slot, k] of heroText) ctx.fillText(M.hero.words[M.hero.word_of_text[k]], fr.pos[2 * slot], fr.pos[2 * slot + 1] + 3);
        ctx.textAlign = "start";
      }
      // the constellation lines fade while the particles are disturbed and come back as they settle
      const show = smooth(0.38, 0.55, p) * (1 - smooth(0.6, 0.7, p)) * (1 - smooth(1.5, 7, energy));
      if (!centroids || show <= 0) return;
      const order = [...Array(8).keys()].sort((a, b) => centroids[b][2] - centroids[a][2]).slice(0, 5);
      ctx.lineWidth = 1.2;
      order.forEach((c, j) => {
        const a = centroids[c], b = centroids[8 + M.cluster_match[c]];
        const t = clamp(show * 1.6 - j * 0.12);
        if (t <= 0) return;
        ctx.strokeStyle = rgb([0.55, 0.96, 0.87], 0.7 * t);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(lerp(a[0], b[0], t), lerp(a[1], b[1], t)); ctx.stroke();
        for (const q of [a, b]) { ctx.beginPath(); ctx.arc(q[0], q[1], 6, 0, 7); ctx.stroke(); }
      });
    },
    hoverable: () => false,
  };
}

// The overview: the two clouds start apart (as they leave the stone), and one map moves the images onto the
// captions. No method steps, only the map; once the clouds have met, the colours show the content of each point.
function overview() {
  const t = { value: 0, start: performance.now() + 900 };
  const MOVE = 0.75, DURATION = 3600;  // the move takes the first three quarters, the colours the rest
  const cI = mean(D.unalignedImage), cT = mean(D.unalignedText), cRef = mean(D.refined), cTx = mean(D.text);
  const ANGLE = (128 * Math.PI) / 180;
  const nearest = (slot) => {
    const k = slot % n, list = slot < n ? D.nnImage : D.nnText, offset = slot < n ? n : 0;
    return [0, 1, 2].map((j) => offset + list[3 * k + j]);
  };
  const imageOf = (slot) => imageUrl("t", M.image_rows[slot]);
  const captionOf = (slot, alt) => (slot < n ? (alt ? altTrain?.[slot] || "" : "") : captionsTrain?.[slot - n]?.[0] || "");
  const restart = (delay) => { t.value = 0; t.start = performance.now() + delay; requestFrame(); };
  return {
    id: "overview", manual: 0,
    onEnter() { restart(900); },
    replay() { restart(300); },
    compute(fr, now) {
      t.value = reducedMotion ? 1 : clamp((now - t.start) / DURATION);
      const e = ease(clamp(t.value / MOVE)), colour = smooth(MOVE + 0.03, 1, t.value), view = viewFor(plotRect());
      // the image cloud turns and moves onto the caption cloud: the map applied a little at a time
      const angle = ANGLE * (1 - e), s = lerp(0.55, 1, e), ca = Math.cos(angle), sa = Math.sin(angle);
      const cx = lerp(cI[0], cRef[0], e), cy = lerp(cI[1], cRef[1], e), tx = lerp(cT[0], cTx[0], e), ty = lerp(cT[1], cTx[1], e);
      const hot = hover.slot >= 0 ? new Set([hover.slot, ...(t.value >= 1 ? nearest(hover.slot) : [])]) : null;
      for (let i = 0; i < N; i++) {
        const k = i % n, image = i < n;
        if (image) {
          const x = D.refined[2 * k] - cRef[0], y = D.refined[2 * k + 1] - cRef[1];
          [fr.pos[2 * i], fr.pos[2 * i + 1]] = view.map(cx + (x * ca - y * sa) * s, cy + (x * sa + y * ca) * s);
        } else {
          [fr.pos[2 * i], fr.pos[2 * i + 1]] = view.map(tx + (D.text[2 * k] - cTx[0]) * s, ty + (D.text[2 * k + 1] - cTx[1]) * s);
        }
        const label = superTrain[i];
        let alpha = lerp(0.72, superAlpha(label, 0.8), colour), scale = 1;
        if (hot) { const on = hot.has(i); alpha = on ? 1 : 0.2; scale = on ? 1.8 : 1; }
        setColor(fr, i, mix3(C.neutral, superColor(label), colour), alpha);
        marker(fr, i, scale);
      }
      this.frame = fr; this.done = t.value >= 1;
      hud(false);
      setStageLabel("DINOv2 B/14 and Qwen3-8B, two halves of MS COCO");
      legend(colour > 0.5 ? "super" : "shapes");
      return t.value < 1;
    },
    overlay(ctx) { if (hover.slot >= 0 && this.done) neighbourLines(ctx, this.frame, nearest(hover.slot)); },
    hoverable: () => true,
    categoryOf: (slot) => superTrain[slot],
    card(slot) {
      const image = slot < n;
      if (!this.done) return image ? { left: [{ img: imageOf(slot), alt: captionOf(slot, true) }], right: [] } : { left: [], right: [{ text: captionOf(slot), main: true }] };
      return neighbourCard(image, slot, nearest(slot), { imageOf, captionOf });
    },
  };
}

function method() {
  // Clustering, read-out and refinement play as timed animations once their step is chosen.
  const clusterTween = { value: 0 };
  const readTween = { value: 0, target: 0 };
  const refineTween = { value: 0, target: 0, last: performance.now() };
  const cI = mean(D.unalignedImage), cT = mean(D.unalignedText), cRef = mean(D.refined), cRo = mean(D.readout), cTx = mean(D.text);
  const ANGLE = (128 * Math.PI) / 180;
  const inverse = new Uint8Array(8);
  M.cluster_match.forEach((t, c) => { inverse[t] = c; });
  const clusterOfText = (k) => inverse[D.clusterText[k]];  // matched clusters share a colour

  function positions(fr, view) {
    const r = ease(readTween.value);                      // read-out: rotate and merge
    const q = refineTween.value;                          // refinement: a short animation
    const angle = ANGLE * (1 - r), s = lerp(0.55, 1, r);
    const ca = Math.cos(angle), sa = Math.sin(angle);
    const cx = lerp(cI[0], cRo[0], r), cy = lerp(cI[1], cRo[1], r);
    for (let k = 0; k < n; k++) {
      const x = lerp(D.refined[2 * k] - cRef[0], D.readout[2 * k] - cRo[0], r), y = lerp(D.refined[2 * k + 1] - cRef[1], D.readout[2 * k + 1] - cRo[1], r);
      let px = cx + (x * ca - y * sa) * s, py = cy + (x * sa + y * ca) * s;
      px = lerp(px, D.refined[2 * k], q); py = lerp(py, D.refined[2 * k + 1], q);
      [fr.pos[2 * k], fr.pos[2 * k + 1]] = view.map(px, py);
    }
    const tx = lerp(cT[0], cTx[0], r), ty = lerp(cT[1], cTx[1], r);
    for (let k = 0; k < n; k++) {
      [fr.pos[2 * (n + k)], fr.pos[2 * (n + k) + 1]] = view.map(tx + (D.text[2 * k] - cTx[0]) * s, ty + (D.text[2 * k + 1] - cTx[1]) * s);
    }
  }
  const superMix = (f) => smooth(2.5, 2.9, f) * refineTween.value;
  function colors(fr, f) {
    const sm = superMix(f);
    const cluster = ease(clusterTween.value) * (1 - sm);
    const hot = hover.slot >= 0 ? new Set([hover.slot, ...(f >= 2 ? neighbours(hover.slot) : [])]) : null;
    for (let i = 0; i < N; i++) {
      const image = i < n, k = i % n;
      const cl = CLUSTER[image ? D.clusterImage[k] : clusterOfText(k)];
      const label = superTrain[i];
      const c = mix3(mix3(C.neutral, cl, cluster), superColor(label), sm);
      let alpha = lerp(0.72, superAlpha(label, 0.8), sm), scale = 1;
      if (hot) { const on = hot.has(i); alpha = on ? 1 : 0.2; scale = on ? 1.8 : 1; }
      setColor(fr, i, c, alpha);
      marker(fr, i, scale);
    }
  }
  function neighbours(slot) {
    const readout = refineTween.value < 0.5, k = slot % n;
    const list = slot < n ? (readout ? D.readoutNnImage : D.nnImage) : (readout ? D.readoutNnText : D.nnText);
    const offset = slot < n ? n : 0;
    return [0, 1, 2].map((j) => offset + list[3 * k + j]);
  }
  const imageOf = (slot) => imageUrl("t", M.image_rows[slot]);
  const captionOf = (slot, alt) => (slot < n ? (alt ? altTrain?.[slot] || "" : "") : captionsTrain?.[slot - n]?.[0] || "");

  return {
    id: "method", manual: 0,
    // the state at a given step, with the tweens where they are (also used by the title's flight)
    state(fr, f) {
      positions(fr, viewFor(plotRect()));
      colors(fr, f);
    },
    compute(fr, now) {
      const f = this.manual;
      this.f = f;
      readTween.target = f >= 2 ? 1 : 0;
      refineTween.target = f >= 3 ? 1 : 0;
      const dt = Math.min(64, now - refineTween.last); refineTween.last = now;
      const before = refineTween.value + readTween.value + clusterTween.value;
      clusterTween.value = reducedMotion ? +(f >= 1) : clamp(clusterTween.value + ((f >= 1 ? 1 : -1) * dt) / 900);
      readTween.value = clamp(readTween.value + (Math.sign(readTween.target - readTween.value) * dt) / 1800);
      // the refinement starts once the read-out has finished
      if (readTween.value >= 1 || refineTween.target === 0) refineTween.value = clamp(refineTween.value + (Math.sign(refineTween.target - refineTween.value) * dt) / 1600);
      if (reducedMotion) { readTween.value = readTween.target; refineTween.value = refineTween.target; }
      this.state(fr, f, now);
      this.frame = fr;
      hud(false);
      setStageLabel("DINOv2 B/14 and Qwen3-8B, two halves of MS COCO");
      legend(superMix(f) > 0.5 ? "super" : "shapes");
      return before !== refineTween.value + readTween.value + clusterTween.value || (f >= 3 && refineTween.value < 1);
    },
    reset() { clusterTween.value = readTween.value = refineTween.value = 0; readTween.target = refineTween.target = 0; refineTween.last = performance.now(); },
    // the first step's frame (two spaces apart, neutral colours) whatever the tweens are doing
    apart(fr) {
      const saved = [clusterTween.value, readTween.value, refineTween.value];
      clusterTween.value = readTween.value = refineTween.value = 0;
      const h = hover.slot; hover.slot = -1;
      this.state(fr, 0);
      [clusterTween.value, readTween.value, refineTween.value] = saved; hover.slot = h;
    },
    done(k) { return k === 1 ? clusterTween.value >= 1 : k === 2 ? readTween.value >= 1 : k === 3 ? refineTween.value >= 1 : true; },
    replay(k) {
      if (k === 0) pinStart = 0;
      if (k === 1) clusterTween.value = 0;
      if (k === 2) { readTween.value = 0; refineTween.value = 0; }
      if (k === 3) refineTween.value = 0;
      refineTween.last = performance.now();
      requestFrame();
    },
    get refined() { return refineTween.value; },
    get readout() { return readTween.value; },
    get clusters() { return clusterTween.value; },
    overlay(ctx) {
      const f = this.f, fr = this.frame;
      const lines = smooth(0.3, 1, this.clusters) * (1 - smooth(0, 0.3, this.readout));  // gone once the read-out starts
      if (lines > 0) {
        const sums = new Float32Array(32), counts = new Float32Array(16);
        for (let i = 0; i < N; i++) {
          const image = i < n, k = i % n, c = image ? D.clusterImage[k] : 8 + clusterOfText(k);
          sums[2 * c] += fr.pos[2 * i]; sums[2 * c + 1] += fr.pos[2 * i + 1]; counts[c]++;
        }
        for (let c = 0; c < 8; c++) {
          const a = [sums[2 * c] / counts[c], sums[2 * c + 1] / counts[c]], b = [sums[2 * (8 + c)] / counts[8 + c], sums[2 * (8 + c) + 1] / counts[8 + c]];
          const t = clamp(lines * 1.4 - c * 0.05);
          ctx.strokeStyle = rgb(CLUSTER[c], 0.85 * t);
          ctx.lineWidth = 1.6;
          const mx = (a[0] + b[0]) / 2, my = Math.min(a[1], b[1]) - 40 - c * 6;
          ctx.beginPath(); ctx.moveTo(a[0], a[1]);
          ctx.quadraticCurveTo(lerp(a[0], mx, t), lerp(a[1], my, t), lerp(a[0], b[0], t), lerp(a[1], b[1], t));
          ctx.stroke();
          for (const [p, image] of [[a, true], [b, false]]) {
            ctx.fillStyle = rgb(CLUSTER[c], t); ctx.strokeStyle = rgb([1, 1, 1], 0.9 * t); ctx.lineWidth = 2;
            ctx.beginPath();
            if (image) ctx.arc(p[0], p[1], 6.5, 0, 7);
            else { ctx.moveTo(p[0], p[1] - 8); ctx.lineTo(p[0] + 7.5, p[1] + 5); ctx.lineTo(p[0] - 7.5, p[1] + 5); ctx.closePath(); }
            ctx.fill(); ctx.stroke();
          }
        }
      }
      linkCallouts(ctx, fr, f >= 3 ? smooth(0.85, 1, this.refined) : 0);
      pinnedCallouts(ctx, fr, f, performance.now());
      if (hover.slot >= 0 && f >= 2) neighbourLines(ctx, fr, neighbours(hover.slot));
    },
    hoverable: () => true,
    categoryOf: (slot) => superTrain[slot],
    card(slot) {
      const image = slot < n;
      if (this.f < 2) {  // before the read-out there is no shared space, so only the point itself
        return image ? { left: [{ img: imageOf(slot), alt: captionOf(slot, true) }], right: [] }
                     : { left: [], right: [{ text: captionOf(slot), main: true }] };
      }
      return neighbourCard(image, slot, neighbours(slot), { imageOf, captionOf });
    },
  };
}

function arrowhead(ctx, from, to, size, color) {
  const ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
  ctx.beginPath(); ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - size * Math.cos(ang - 0.42), to[1] - size * Math.sin(ang - 0.42));
  ctx.lineTo(to[0] - size * Math.cos(ang + 0.42), to[1] - size * Math.sin(ang + 0.42));
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}

function neighbourLines(ctx, fr, list) {
  const a = [fr.pos[2 * hover.slot], fr.pos[2 * hover.slot + 1]];
  ctx.setLineDash([2, 3]); ctx.lineWidth = 1; ctx.strokeStyle = rgb(C.ink, 0.55);
  for (const j of list) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(fr.pos[2 * j], fr.pos[2 * j + 1]); ctx.stroke(); }
  ctx.setLineDash([]);
}

// The first step shows one image and one caption on screen, to show that the points can be inspected.
let pinStart = 0;
function pinnedCallouts(ctx, fr, f, now) {
  const show = hover.slot < 0 && f === 0 && chapter === "method" && SNAP.t >= 1;
  if (!show) { hideCallouts("pin"); pinStart = 0; return; }
  if (!pinStart) pinStart = now + 250;  // after the points have arrived
  const e = reducedMotion ? 1 : ease(clamp((now - pinStart) / 700));
  if (e < 1) bump(60);
  const image = 0, caption = n + D.nnImage[0];
  const a = [fr.pos[2 * image], fr.pos[2 * image + 1]], b = [fr.pos[2 * caption], fr.pos[2 * caption + 1]];
  const rect = plotRect(), size = isMobile() ? 72 : 96;
  const ix = clamp(a[0] - size - 40, rect.x, rect.x + rect.w - size), iy = clamp(a[1] - size - 30, rect.y, rect.y + rect.h - size);
  showCallout("pin-image", { x: ix, y: iy, html: `<img src="${imageUrl("t", M.image_rows[image])}" alt="${escapeHtml(altTrain?.[image] || "")}" style="width:${size}px;height:${size}px">` });
  const cw = isMobile() ? 170 : 220;
  const cx = clamp(b[0] + 30, rect.x, rect.x + rect.w - cw), cy = clamp(b[1] + 34, rect.y, rect.y + rect.h - 60);
  showCallout("pin-caption", { x: cx, y: cy, html: `<div class="hc-cap" style="max-width:${cw}px">${escapeHtml(captionsTrain?.[D.nnImage[0]]?.[0] || "")}</div>` });
  for (const key of ["pin-image", "pin-caption"]) {
    const el = callouts.get(key);
    el.style.opacity = e; el.style.transform = `translateY(${(1 - e) * 10}px) scale(${0.96 + 0.04 * e})`;
  }
  // the leader lines grow from the points to the cards
  ctx.strokeStyle = rgb(C.ink, 0.5); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(lerp(a[0], ix + size / 2, e), lerp(a[1], iy + size, e)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(b[0], b[1]); ctx.lineTo(lerp(b[0], cx, e), lerp(b[1], cy + 12, e)); ctx.stroke();
}

// A card tied to its own point: it takes the first free spot around the point (outward first), inside the plot
// and clear of everything in ``taken``; a leader runs from the point to the card's nearest edge.
function placeCard(ctx, key, html, p, rect, taken, out, alpha) {
  showCallout(key, { x: -9999, y: -9999, html });
  const el = callouts.get(key), w = el.offsetWidth, h = el.offsetHeight, D = 40;
  const free = (b) => b.x >= rect.x && b.y >= rect.y && b.x + b.w <= rect.x + rect.w && b.y + b.h <= rect.y + rect.h
    && !taken.some((o) => b.x < o.x + o.w + 6 && b.x + b.w + 6 > o.x && b.y < o.y + o.h + 6 && b.y + b.h + 6 > o.y);
  let box = null;
  for (const [dx, dy] of [[out, -1], [out, 0], [out, 1], [0, -1], [0, 1], [-out, -1], [-out, 0], [-out, 1]]) {
    const d = dx && dy ? D * 0.8 : D;
    const b = { x: p[0] + dx * d - (dx < 0 ? w : dx === 0 ? w / 2 : 0), y: p[1] + dy * d - (dy < 0 ? h : dy === 0 ? h / 2 : 0), w, h };
    if (free(b)) { box = b; break; }
  }
  if (!box) {  // no room right around the point: the nearest free spot anywhere in the plot, else on top of it
    let best = Infinity;
    for (let x = rect.x; x <= rect.x + rect.w - w; x += 10) for (let y = rect.y; y <= rect.y + rect.h - h; y += 10) {
      const b = { x, y, w, h }, d = Math.hypot(clamp(p[0], x, x + w) - p[0], clamp(p[1], y, y + h) - p[1]);
      if (d > 0 && d < best && free(b)) { best = d; box = b; }
    }
  }
  box ??= { x: p[0] + out * D, y: p[1] - h - D * 0.8, w, h };
  box.x = clamp(box.x, rect.x, rect.x + rect.w - w); box.y = clamp(box.y, rect.y, rect.y + rect.h - h);
  el.style.left = `${box.x}px`; el.style.top = `${box.y}px`; el.style.opacity = alpha;
  taken.push(box);
  ctx.strokeStyle = rgb(C.ink, 0.6 * alpha); ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(clamp(p[0], box.x, box.x + w), clamp(p[1], box.y, box.y + h)); ctx.stroke();
  return box;
}
// A ring around a point in its own shape: a circle for an image, a triangle for a caption.
function ring(ctx, p, image, alpha, r = 9) {
  ctx.strokeStyle = rgb(C.ink, 0.9 * alpha); ctx.lineWidth = 1.6;
  ctx.beginPath();
  if (image) ctx.arc(p[0], p[1], r * 0.8, 0, 7);
  else { ctx.moveTo(p[0], p[1] - r); ctx.lineTo(p[0] + r * 0.95, p[1] + r * 0.7); ctx.lineTo(p[0] - r * 0.95, p[1] + r * 0.7); ctx.closePath(); }
  ctx.stroke();
}

// After the refinement: a few images and the caption each landed next to. They were never a pair (the method saw
// no pairs, and the caption is not the image's own), so each card hangs from its own point.
function linkCallouts(ctx, fr, glow) {
  if (glow <= 0 || hover.slot >= 0 || chapter !== "method") { hideCallouts("link"); return; }
  const rect = plotRect(), cx = rect.x + rect.w / 2;
  const small = isMobile() || rect.h < 360;
  const links = M.links.slice(0, rect.h < 300 ? 1 : small ? 2 : 3).map(([i, t]) => [[fr.pos[2 * i], fr.pos[2 * i + 1]], [fr.pos[2 * (n + t)], fr.pos[2 * (n + t) + 1]], i, t]);
  const taken = links.flatMap(([a, b]) => [a, b].map((p) => ({ x: p[0] - 9, y: p[1] - 9, w: 18, h: 18 })));
  const size = small ? 52 : 76;
  links.forEach(([a, b, i, t], j) => {
    const e = clamp(glow * 1.5 - j * 0.2), out = (a[0] + b[0]) / 2 < cx ? -1 : 1;
    placeCard(ctx, `link-image-${j}`, `<img src="${imageUrl("t", M.image_rows[i])}" alt="${escapeHtml(altTrain?.[i] || "")}" style="width:${size}px;height:${size}px">`, a, rect, taken, out, e);
    placeCard(ctx, `link-caption-${j}`, `<div class="hc-cap" style="width:${small ? 140 : 200}px">${escapeHtml(captionsTrain?.[t]?.[0] || "")}</div>`, b, rect, taken, out, e);
    ring(ctx, a, true, e); ring(ctx, b, false, e);
  });
}

// A morph between validation layouts (data units, slot order) on a timer.
class Morph {
  constructor() { this.from = null; this.to = null; this.start = 0; this.duration = 900; }
  set(target, now = performance.now(), immediate = false) {
    if (!this.to || immediate || reducedMotion) { this.to = target; this.from = target; this.start = 0; return; }
    this.from = this.current(now); this.to = target; this.start = now;
    bump(this.duration + 50);
  }
  current(now) {
    const t = this.start ? ease(clamp((now - this.start) / this.duration)) : 1;
    if (t >= 1) return this.to;
    const out = new Float32Array(this.to.length);
    for (let i = 0; i < out.length; i++) out[i] = lerp(this.from[i], this.to[i], t);
    return out;
  }
}

// Validation block -> slot order (data units).
function toSlots(blockData) {
  const out = new Float32Array(2 * N);
  for (let k = 0; k < n; k++) {
    const vi = D.toValImage[k], vt = D.toValText[k];
    out[2 * k] = blockData.pos[2 * vi]; out[2 * k + 1] = blockData.pos[2 * vi + 1];
    out[2 * (n + k)] = blockData.pos[2 * (n + vt)]; out[2 * (n + k) + 1] = blockData.pos[2 * (n + vt) + 1];
  }
  return out;
}
let invImage, invText;  // validation index -> slot

function valFrame(fr, positions, info, view, { alpha = 0.8 } = {}) {
  const hot = hover.slot >= 0 && info ? new Set(valHoverSlots(hover.slot, info)) : null;
  for (let i = 0; i < N; i++) {
    [fr.pos[2 * i], fr.pos[2 * i + 1]] = view.map(positions[2 * i], positions[2 * i + 1]);
    const label = superOf(i, "val");
    let a = superAlpha(label, alpha), scale = 1;
    if (hot) { const on = hot.has(i); a = on ? 1 : 0.18; scale = on ? 1.8 : 1; }
    setColor(fr, i, superColor(label), a);
    marker(fr, i, scale);
  }
}
const valIndex = (slot) => (slot < n ? D.toValImage[slot] : D.toValText[slot - n]);
function valHoverSlots(slot, info) {
  const image = slot < n, v = valIndex(slot);
  const nn = [0, 1, 2].map((j) => info.nn[3 * (image ? v : n + v) + j]);
  return [slot, image ? invText[v] : invImage[v], ...nn.map((j) => (image ? invText[j] : invImage[j]))];
}
function valCard(slot, info) {
  const image = slot < n, v = valIndex(slot);
  const nn = [0, 1, 2].map((j) => info.nn[3 * (image ? v : n + v) + j]);
  return neighbourCard(image, v, nn, {
    imageOf: (j) => imageUrl("v", M.val_rows[j]),
    captionOf: (j) => captionsVal?.[j]?.[0] || "",
    truth: { index: v, slot: image ? invText[v] : invImage[v] },
    closer: info.closer[image ? v : n + v],
  });
}

function drawTruth(ctx, fr, card) {
  if (!card || card.truth === undefined || hover.slot < 0) return;
  const a = [fr.pos[2 * hover.slot], fr.pos[2 * hover.slot + 1]], b = [fr.pos[2 * card.truth], fr.pos[2 * card.truth + 1]];
  ctx.strokeStyle = rgb(C.ink, 0.85); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  ctx.beginPath(); ctx.arc(b[0], b[1], 8, 0, 7); ctx.stroke();
  const label = `${pct(card.closer)}% of ${card.closerNoun} are closer`;
  const long = Math.hypot(b[0] - a[0], b[1] - a[1]) > 90;
  let mx = long ? (a[0] + b[0]) / 2 : b[0], my = long ? (a[1] + b[1]) / 2 : b[1] - 8;
  ctx.font = '600 13px "Work Sans", system-ui, sans-serif';
  const w = ctx.measureText(label).width + 12;
  // keep the label clear of the cards: try above the true match, then below it, then left of the cards
  const cards = [hoverLeft, hoverRight].filter((c) => c.style.display !== "none").map((c) => c.getBoundingClientRect());
  const hits = (x, y) => cards.some((r) => x + w / 2 > r.left - 4 && x - w / 2 < r.right + 4 && y > r.top - 4 && y - 22 < r.bottom + 4);
  if (hits(mx, my)) {
    if (!hits(b[0], b[1] + 34)) { mx = b[0]; my = b[1] + 34; }
    else if (cards.length) mx = Math.min(...cards.map((r) => r.left)) - 8 - w / 2;
  }
  const rect = plotRect();  // never over the panel
  mx = clamp(mx, rect.x + w / 2, rect.x + rect.w - w / 2);
  ctx.fillStyle = rgb(C.paper, 0.94);
  ctx.beginPath(); ctx.roundRect(mx - w / 2, my - 22, w, 19, 2); ctx.fill();
  ctx.fillStyle = rgb(C.ink); ctx.textAlign = "center";
  ctx.fillText(label, mx, my - 8);
  ctx.textAlign = "start";
}

const comboLabel = (c) => `${M.labels.vision[c.vision]}, ${M.labels.language[c.language]}, ${M.labels.dataset[c.dataset]}`;

// Replaying a results chapter shows the alignment itself: the two spaces start apart, as in the method's first
// step, and the map brings them together.
function replayFromApart() {
  scenes.method.apart(SNAP.frame);
  SNAP.t = 0; SNAP.last = performance.now();
  requestFrame();
}

function results() {
  const morph = new Morph();
  let combo = M.combos[M.default_combo], info = null, pending = null;
  const selects = { vision: $("sel-vision"), language: $("sel-language"), dataset: $("sel-dataset") };
  const order = { vision: Object.keys(M.labels.vision), language: Object.keys(M.labels.language), dataset: Object.keys(M.labels.dataset) };
  for (const [key, select] of Object.entries(selects)) {
    for (const value of order[key]) {
      if (!M.combos.some((c) => c[key] === value)) continue;
      select.add(new Option(M.labels[key][value], value));
    }
    select.value = combo[key];
    select.addEventListener("change", () => {
      const c = M.combos.find((x) => x.vision === selects.vision.value && x.language === selects.language.value && x.dataset === selects.dataset.value);
      if (c) load(c);
    });
  }
  async function load(c, immediate = false) {
    pending = c;
    const data = await block(`data/combo/${c.id}.bin`);
    if (pending !== c) return;
    combo = c; info = data;
    morph.set(toSlots(data), performance.now(), immediate);
    setMetric(); bump();
  }
  const setMetric = () => { $("metric-value").textContent = fmt(combo.foscttm); $("metric-sub").textContent = `± ${fmt(combo.foscttm_std)} over 5 seeds`; };
  let hist = null, histKey = "", histWidth = 0;
  function updateHistogram() {
    if (!info) return;
    const host = $("hist");
    if (histWidth !== host.clientWidth) { hist = null; histKey = ""; host.innerHTML = ""; histWidth = host.clientWidth; }
    if (!hist && host.clientWidth) hist = histogram(host);
    const marked = hover.slot >= 0 && hover.slot < n ? info.closer[D.toValImage[hover.slot]] : null;
    const key = `${combo.id}:${marked}`;
    if (hist && key !== histKey) { histKey = key; hist.update(info.closer.subarray(0, n), marked); }
  }
  return {
    id: "results", manual: 0,
    ready: () => load(combo, true),
    state(fr, f, now) {
      if (!info) { scenes.method.state(fr, 3, now); return; }
      valFrame(fr, morph.current(now), info, viewFor(plotRect()));
    },
    compute(fr, now) {
      this.state(fr, 0, now);
      this.frame = fr;
      setStageLabel(comboLabel(combo));
      legend("super");
      hud(true); setMetric(); updateHistogram();
      return false;
    },
    replay: replayFromApart,
    overlay(ctx) { if (info && hover.slot >= 0) drawTruth(ctx, this.frame, valCard(hover.slot, info)); },
    hoverable() { return !!info; },
    card: (slot) => valCard(slot, info),
    categoryOf: (slot) => superOf(slot, "val"),
  };
}

function pairs() {
  const slider = $("pairs-slider"), value = $("pairs-value");
  const t2iSlider = $("t2i-slider"), t2iValue = $("t2i-value");
  const morph = new Morph();
  let index = 0, info = null, caption = 0, chosenAt = 0;
  const mapped = { from: null, to: null, start: 0, shown: 0 };
  // The known pairs of the current map: drawn over the cloud, the first few with their image and caption.
  let known = null, knownStart = 0, knownHover = -1, knownScreen = new Float32Array(0), knownCount = 0;
  let spark = null, sparkWidth = 0;
  const sparkMark = () => {
    const host = $("pairs-spark");
    if (sparkWidth !== host.clientWidth) { spark = null; host.innerHTML = ""; sparkWidth = host.clientWidth; }
    if (!spark && host.clientWidth) spark = sparkline(host, M.fewpair_curve);
    spark?.mark(M.fewpair_curve.num_pairs.indexOf(M.fewpair[index].num_pairs));
  };
  slider.max = M.fewpair.length - 1;
  async function load(i, immediate = false) {
    index = i;
    const entry = M.fewpair[i];
    value.textContent = entry.num_pairs;
    sparkMark();
    const data = await block(entry.file);
    if (index !== i) return;
    info = data;
    morph.set(toSlots(data), performance.now(), immediate);
    knownStart = performance.now(); knownHover = -1; knownCount = 0;
    setMetric(); bump();
  }
  const setMetric = () => {
    const e = M.fewpair[index];
    $("metric-value").textContent = fmt(e.foscttm);
    $("metric-sub").textContent = `with ${e.num_pairs} known pair${e.num_pairs === 1 ? "" : "s"}`;
  };
  slider.addEventListener("input", () => load(+slider.value));

  const choose = (i) => {
    caption = (i + M.t2i.captions.length) % M.t2i.captions.length; chosenAt = performance.now();
    $("cap-text").textContent = M.t2i.captions[caption];
    bump(2600); preload();
  };
  $("cap-text").textContent = M.t2i.captions[0];
  $("cap-prev").addEventListener("click", () => choose(caption - 1));
  $("cap-next").addEventListener("click", () => choose(caption + 1));
  t2iSlider.max = M.t2i.pairs.length - 1;
  t2iSlider.addEventListener("input", () => { t2iValue.textContent = M.t2i.pairs[+t2iSlider.value]; bump(900); preload(); });
  const preload = () => { new Image().src = `data/img/g${M.t2i.pairs[+t2iSlider.value]}_${caption}.webp`; };

  const m = () => M.n_t2i;
  const scratch = makeFrame(N), t2iFrame = makeFrame(N);
  function t2iState(fr) {
    const view = viewFor(plotRect());
    const nnSlots = new Set();
    const p = +t2iSlider.value;
    for (let j = 0; j < 3; j++) nnSlots.add(T2I.nn[3 * (p * M.t2i.captions.length + caption) + j]);
    for (let i = 0; i < N; i++) {
      const image = i < n, k = i % n, kk = k < m() ? k : k % m();
      const src = image ? T2I.vision : T2I.language;
      [fr.pos[2 * i], fr.pos[2 * i + 1]] = view.map(src[2 * kk], src[2 * kk + 1]);
      const label = superT2I[kk];
      let a = k < m() ? superAlpha(label, 0.32) : 0, scale = 1;
      if (image && nnSlots.has(k)) { a = 0.95; scale = 1.6; }
      if (hover.slot >= 0 && k < m()) { if (i === hover.slot) { a = 1; scale = 1.9; } else a *= 0.6; }
      setColor(fr, i, superColor(label), a);
      marker(fr, i, scale);
    }
  }
  // The move to the text-to-image setting plays as a timed animation; the map arrow and the generated image
  // follow when the clouds have arrived.
  const toT2I = { value: 0, last: performance.now() };
  return {
    id: "pairs", manual: 0,
    async ready() {
      await load(0, true);
      known = await loadJSON("data/fewpair/known.json").catch(() => null);
      requestFrame();
    },
    onEnter(step) { toT2I.value = step; chosenAt = performance.now() + 900; knownStart = performance.now() + 500; knownHover = -1; },
    state(fr, f, now) {
      if (!info || !T2I) { scenes.results.state(fr, 0, now); return; }
      const e = ease(toT2I.value);
      if (e <= 0) { valFrame(fr, morph.current(now), info, viewFor(plotRect())); return; }
      valFrame(scratch, morph.current(now), null, viewFor(plotRect()));
      t2iState(t2iFrame);
      blend(fr, scratch, t2iFrame, e);
    },
    compute(fr, now) {
      const f = this.manual;
      this.f = f;
      const dt = Math.min(64, now - toT2I.last); toT2I.last = now;
      const before = toT2I.value, target = f >= 1 && T2I ? 1 : 0;
      toT2I.value = reducedMotion ? target : clamp(toT2I.value + (Math.sign(target - toT2I.value) * dt) / 700);
      if (before < 1 && toT2I.value >= 1) chosenAt = now;  // start the map arrow once the clouds are in place
      this.state(fr, f, now);
      this.frame = fr;
      const t2i = toT2I.value > 0.5;
      this.t2i = t2i && !!T2I;
      setStageLabel(t2i ? "Captions: MPNet (right). Images: the RAE's DINOv2 B/14 (left)." : "DINOv2 B/14, Qwen3 gen, MS COCO");
      legend(!t2i && M.fewpair[index].num_pairs > 0 ? "known" : "super");
      hud(!t2i);
      if (!t2i) { setMetric(); sparkMark(); }
      if (!t2i) hideCallouts("t2i");
      if (f >= 1) scenes.geometry.prefetch();
      return toT2I.value !== before || (target === 1 && toT2I.value < 1);
    },
    replay(step) {
      if (step >= 1) { chosenAt = performance.now(); toT2I.value = 0; bump(2600); }
      else replayFromApart();
    },
    overlay(ctx, now) {
      if (!this.t2i) {
        if (info && toT2I.value === 0) drawKnown(ctx, now); else { knownCount = 0; hideCallouts("known"); }
        if (info && hover.slot >= 0) drawT2IOff(ctx);
        return;
      }
      hideCallouts("known");
      if (toT2I.value < 1) { hideCallouts("t2i"); return; }
      drawT2I(ctx, now);
    },
    hoverable() { return !!info && (toT2I.value === 0 || toT2I.value === 1); },
    // a known pair with an image and a caption near (x, y), or -1; hovering one shows its two cards
    pickKnown(x, y, radius) {
      const entry = knownEntry();
      if (!entry || this.t2i) return -1;
      let best = -1, bestD = radius * radius;
      for (let j = 0; j < Math.min(knownCount, entry.rows.length); j++) {
        for (const o of [0, 2]) {
          const dx = knownScreen[4 * j + o] - x, dy = knownScreen[4 * j + o + 1] - y, d = dx * dx + dy * dy;
          if (d < bestD) { bestD = d; best = j; }
        }
      }
      return best;
    },
    hoverKnown(j) { if (j !== knownHover) { knownHover = j; requestFrame(); } return j; },
    categoryOf(slot) { return this.t2i ? superT2I[slot % n] : superOf(slot, "val"); },
    card(slot) {
      if (!this.t2i) return valCard(slot, info);
      const k = slot % n;
      if (k >= m()) return null;
      return slot < n ? { left: [{ img: imageUrl("t", M.t2i_rows[k]), alt: captionsT2I?.[k]?.[0] || "", label: "image" }], right: [] }
                      : { left: [], right: [{ text: captionsT2I?.[k]?.[0] || "", label: "caption", main: true }] };
    },
  };

  function drawT2IOff(ctx) { drawTruth(ctx, scenes.pairs.frame, valCard(hover.slot, info)); }

  function knownEntry() { const k = M.fewpair[index].num_pairs; return k > 0 ? known?.[k] : null; }
  // Each known pair: its image and its caption, outlined, in the colour of the image's content, joined by a line
  // (these pairs are given to the method). They fade in once the cloud has moved to the new map.
  function drawKnown(ctx, now) {
    const entry = knownEntry();
    if (!entry || SNAP.t < 1) { knownCount = 0; hideCallouts("known"); return; }
    const count = entry.pos.length / 4, view = viewFor(plotRect());
    const t = reducedMotion ? 1 : ease(clamp((now - knownStart - 500) / 700));
    if (t < 1) bump(60);
    if (knownScreen.length < 4 * count) knownScreen = new Float32Array(4 * count);
    for (let j = 0; j < count; j++) {
      [knownScreen[4 * j], knownScreen[4 * j + 1]] = view.map(entry.pos[4 * j], entry.pos[4 * j + 1]);
      [knownScreen[4 * j + 2], knownScreen[4 * j + 3]] = view.map(entry.pos[4 * j + 2], entry.pos[4 * j + 3]);
    }
    knownCount = count;
    // a few pairs stand out; hundreds are drawn lighter, so that they do not hide the cloud they sit in
    const tier = count <= 20 ? { r: 4.6, line: 1.6, lineA: 0.85, edge: 1.5, edgeA: 0.95, fillA: 1 }
      : count <= 100 ? { r: 4, line: 1.2, lineA: 0.7, edge: 1.2, edgeA: 0.85, fillA: 1 }
      : { r: 2.8, line: 0.8, lineA: 0.3, edge: 0.8, edgeA: 0.5, fillA: 0.75 };
    const dim = (hover.slot >= 0 || knownHover >= 0 ? 0.35 : 1) * t, r = tier.r * pointScale;
    ctx.lineWidth = tier.line;
    ctx.strokeStyle = rgb(C.ink, tier.lineA * dim);
    ctx.beginPath();
    for (let j = 0; j < count; j++) { ctx.moveTo(knownScreen[4 * j], knownScreen[4 * j + 1]); ctx.lineTo(knownScreen[4 * j + 2], knownScreen[4 * j + 3]); }
    ctx.stroke();
    for (let j = 0; j < count; j++) {
      const fill = SUPER[+entry.super[j]] || C.neutral;
      const [ax, ay, bx, by] = knownScreen.subarray(4 * j, 4 * j + 4);
      ctx.fillStyle = rgb(fill, tier.fillA * dim); ctx.strokeStyle = rgb(C.ink, tier.edgeA * dim); ctx.lineWidth = tier.edge;
      ctx.beginPath(); ctx.arc(ax, ay, r * 0.85, 0, 7); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(bx, by - r * 1.15); ctx.lineTo(bx + r, by + r * 0.75); ctx.lineTo(bx - r, by + r * 0.75); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    // the hovered pair: its image at the image point, its caption at the caption point
    if (knownHover < 0 || knownHover >= entry.rows.length || hover.slot >= 0) { hideCallouts("known"); return; }
    const j = knownHover, a = [knownScreen[4 * j], knownScreen[4 * j + 1]], b = [knownScreen[4 * j + 2], knownScreen[4 * j + 3]];
    const rect = plotRect(), small = isMobile() || rect.h < 360, size = small ? 64 : 96;
    const taken = [a, b].map((p) => ({ x: p[0] - 10, y: p[1] - 10, w: 20, h: 20 })), out = a[0] < rect.x + rect.w / 2 ? -1 : 1;
    placeCard(ctx, "known-image", `<img src="${imageUrl("t", entry.rows[j])}" alt="${escapeHtml(entry.captions[j])}" style="width:${size}px;height:${size}px">`, a, rect, taken, out, 1);
    placeCard(ctx, "known-caption", `<div class="hc-cap" style="width:${small ? 160 : 220}px"><small>known pair</small>${escapeHtml(entry.captions[j])}</div>`, b, rect, taken, out, 1);
    ring(ctx, a, true, 1, 11); ring(ctx, b, false, 1, 12);
  }

  // The caption (a ring), the map (an arc into the image space) and the RAE decoder (an arrow to the image that
  // it generates, shown next to the place where the caption lands).
  function drawT2I(ctx, now) {
    const view = viewFor(plotRect()), rect = plotRect();
    const p = +t2iSlider.value, c = caption, P = M.t2i.captions.length;
    const start = view.map(T2I.captions[2 * c], T2I.captions[2 * c + 1]);
    const endData = [T2I.mapped[2 * (p * P + c)], T2I.mapped[2 * (p * P + c) + 1]];
    if (mapped.shown !== p) { mapped.from = mapped.to || endData; mapped.shown = p; mapped.start = now; }
    const mt = mapped.start ? ease(clamp((now - mapped.start) / 800)) : 1;
    const from = mapped.from || endData;
    mapped.to = [lerp(from[0], endData[0], mt), lerp(from[1], endData[1], mt)];
    const end = view.map(mapped.to[0], mapped.to[1]);
    const t = reducedMotion ? 1 : clamp((now - chosenAt) / 1300);
    const pulse = 1 + 0.25 * Math.sin(now / 240) * (1 - smooth(0, 0.3, t));
    ctx.strokeStyle = rgb(C.ink); ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(start[0], start[1], 8 * pulse, 0, 7); ctx.stroke();
    ctx.fillStyle = rgb(C.ink); ctx.beginPath(); ctx.arc(start[0], start[1], 2.5, 0, 7); ctx.fill();
    const a = smooth(0.2, 0.65, t);
    const mx = (start[0] + end[0]) / 2, my = Math.max(rect.y + 10, Math.min(start[1], end[1]) - 90);
    if (a > 0) {
      ctx.strokeStyle = rgb(C.ink, 0.8); ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let s = 0; s <= 40 * a; s++) {
        const u = s / 40, x = (1 - u) * (1 - u) * start[0] + 2 * (1 - u) * u * mx + u * u * end[0], y = (1 - u) * (1 - u) * start[1] + 2 * (1 - u) * u * my + u * u * end[1];
        s ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
      ctx.font = '600 14px "Work Sans", system-ui, sans-serif'; ctx.fillStyle = rgb(C.ink); ctx.textAlign = "center";
      ctx.fillText("our map", mx, (start[1] + end[1]) / 4 + my / 2 - 9); ctx.textAlign = "start";
    }
    if (a > 0.98) {  // one arrowhead, at the image end
      const u = 0.94, px = (1 - u) * (1 - u) * start[0] + 2 * (1 - u) * u * mx + u * u * end[0], py = (1 - u) * (1 - u) * start[1] + 2 * (1 - u) * u * my + u * u * end[1];
      arrowhead(ctx, [px, py], end, 17, rgb(C.ink, 0.9));
    }
    // the generated image: below the place where the caption lands, or above it when there is no room
    const g = smooth(0.62, 1, t);
    const cramped = rect.h < 340;
    const size = Math.round(cramped ? clamp(rect.h * 0.4, 64, 112) : clamp(Math.min(rect.w, rect.h) * 0.36, 96, 190));
    let ix, iy;
    if (cramped) {  // a lower corner of the plot (the map's arc goes over the top), on the side of the image cloud
      ix = end[0] < start[0] ? rect.x : rect.x + rect.w - size;
      iy = end[1] > rect.y + rect.h * 0.6 ? rect.y : rect.y + rect.h - size - 16;
    } else {
      ix = clamp(end[0] - size / 2, rect.x, rect.x + rect.w - size);
      iy = end[1] + 56 + size < rect.y + rect.h ? end[1] + 56 : end[1] - 56 - size;
      iy = clamp(iy, rect.y, rect.y + rect.h - size);
    }
    showCallout("t2i-image", g > 0.02 ? { x: ix, y: iy, html: `<figure class="gen"><img src="data/img/g${M.t2i.pairs[p]}_${c}.webp" alt="Generated for: ${escapeHtml(M.t2i.captions[c])}" style="width:${size}px;height:${size}px"><figcaption>${cramped ? "RAE decoder, " : ""}${M.t2i.pairs[p]} known pairs</figcaption></figure>` } : null);
    if (g > 0.02) callouts.get("t2i-image").style.opacity = g;
    if (g > 0) {
      const below = iy > end[1];
      const tx = cramped ? (ix < end[0] ? ix + size + 6 : ix - 6) : ix + size / 2, ty = cramped ? iy + size / 2 : below ? iy - 6 : iy + size + 6;
      ctx.strokeStyle = rgb(C.ink, 0.8 * g); ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(end[0], end[1]); ctx.lineTo(lerp(end[0], tx, g), lerp(end[1], ty, g)); ctx.stroke(); ctx.setLineDash([]);
      if (g > 0.95) arrowhead(ctx, end, [tx, ty], 13, rgb(C.ink, 0.85));
      ctx.font = '700 13px "Work Sans", system-ui, sans-serif';
      if (!cramped) {  // on small plots the image's caption names the decoder instead
        const label = "RAE decoder", w = ctx.measureText(label).width + 14, lx = (end[0] + tx) / 2 + w / 2 + 12, ly = (end[1] + ty) / 2;
        ctx.fillStyle = rgb(C.paper, 0.95 * g); ctx.beginPath(); ctx.roundRect(lx - w / 2, ly - 11, w, 22, 2); ctx.fill();
        ctx.fillStyle = rgb(C.ink, g); ctx.textAlign = "center"; ctx.fillText(label, lx, ly + 4.5); ctx.textAlign = "start";
      }
    }
    showCallout("t2i-caption", hover.slot >= 0 || cramped ? null : { x: clamp(start[0] + 14, rect.x, rect.x + rect.w - 230), y: clamp(start[1] - 60, rect.y, rect.y + rect.h - 50), html: `<div class="hc-cap main" style="max-width:220px"><small>caption</small>${escapeHtml(M.t2i.captions[c])}</div>` });
    if (t < 1 || mt < 1) bump(50);
  }
}

function geometry() {
  const slider = $("cka-slider");
  const sorted = M.combos.filter((c) => c.cka != null).sort((a, b) => a.cka - b.cka);
  slider.max = sorted.length - 1;
  const morph = new Morph();
  let index = sorted.findIndex((c) => c.id === M.default_combo), info = null, chart = null, chartWidth = 0, linesStart = 0;
  slider.value = index;
  const scatter = () => {
    const host = $("cka-scatter");
    if (chartWidth !== host.clientWidth) { chart = null; host.innerHTML = ""; chartWidth = host.clientWidth; }
    if (!chart && host.clientWidth) {
      chart = ckaScatter(host, sorted, M.vl_fit, { onPick: (i) => { slider.value = i; load(i); } });
      chart.highlight(index);
      chart.showTrend(true);
    }
    return chart;
  };
  async function load(i, immediate = false) {
    index = i;
    const c = sorted[i];
    $("cka-value").textContent = c.cka.toFixed(2);
    $("cka-combo").textContent = comboLabel(c);
    scatter()?.highlight(i);
    const data = await block(`data/combo/${c.id}.bin`);
    if (index !== i) return;
    info = data;
    morph.set(toSlots(data), performance.now(), immediate);
    setMetric();
    bump();
  }
  const setMetric = () => { const c = sorted[index]; $("metric-value").textContent = fmt(c.foscttm); $("metric-sub").textContent = `± ${fmt(c.foscttm_std)}, CKA ${c.cka.toFixed(2)}`; };
  slider.addEventListener("input", () => load(+slider.value));
  return {
    id: "geometry", manual: 0,
    ready: () => load(index, true),
    prefetch() { if (!this.fetched) { this.fetched = true; sorted.forEach((c) => fetchOnce(`data/combo/${c.id}.bin`)); } },
    state(fr, f, now) {
      if (!info) { scenes.results.state(fr, 0, now); return; }
      valFrame(fr, morph.current(now), info, viewFor(plotRect()));
    },
    compute(fr, now) {
      this.state(fr, 0, now);
      this.frame = fr;
      hud(true); setMetric();
      scatter();
      setStageLabel(comboLabel(sorted[index]));
      legend("super");
      return false;
    },
    replay() { linesStart = 0; replayFromApart(); },
    overlay(ctx) {
      // Thin lines from a fixed set of images to their true captions: short when the alignment is good. They
      // grow from the images once the move into this chapter has finished.
      const fr = this.frame, now = performance.now();
      const arrived = SNAP.t >= 1;
      if (!arrived) linesStart = 0; else if (!linesStart) linesStart = now;
      const e = arrived ? (reducedMotion ? 1 : ease(clamp((now - linesStart) / 900))) : 0;
      if (arrived && e < 1) bump(60);
      if (info && e > 0 && hover.slot < 0) {
        ctx.strokeStyle = rgb(C.ink, 0.35); ctx.lineWidth = 1;
        TRUTH_SAMPLE.forEach((slot, k) => {
          const t = invText[D.toValImage[slot]];
          const g = clamp(e * 1.4 - (k / TRUTH_SAMPLE.length) * 0.4);  // a slight stagger
          if (g <= 0) return;
          const a = [fr.pos[2 * slot], fr.pos[2 * slot + 1]];
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(lerp(a[0], fr.pos[2 * t], g), lerp(a[1], fr.pos[2 * t + 1], g)); ctx.stroke();
        });
      }
      if (info && hover.slot >= 0) drawTruth(ctx, fr, valCard(hover.slot, info));
    },
    hoverable() { return !!info; },
    card: (slot) => valCard(slot, info),
    categoryOf: (slot) => superOf(slot, "val"),
  };
}
let TRUTH_SAMPLE = [];

// Other modalities and specialized methods: the cloud gathers into the middle and gives way to the chart of all
// pairs of modalities, which takes the plot's place.
function chartScene(id) {
  return {
    id, manual: 0,
    compute(fr) {
      const r = plotRect(), cx = r.x + r.w / 2, cy = r.y + r.h / 2;
      for (let i = 0; i < N; i++) { fr.pos[2 * i] = cx; fr.pos[2 * i + 1] = cy; fr.col[4 * i + 3] = 0; }
      this.frame = fr;
      hud(false); setStageLabel(""); legend(null);
      if (id === "benchmarks" && !$("bench-plot").childElementCount) { benchTable($("bench-plot"), M.benchmarks); benchTable($("bench-stage"), M.benchmarks); }
      return false;
    },
    replay() { const host = $("domain-plot"); host.classList.remove("show"); void host.offsetWidth; host.classList.add("show"); },
    hoverable: () => false,
  };
}
let figureKey = "";
function sizeFigure() {
  const host = $("domain-plot"), bench = $("bench-stage");
  // other modalities: the chart; specialized methods: the table on wide screens, the chart dimmed above the sheet
  const tableHere = chapter === "benchmarks" && !isMobile();
  const on = chapter === "domains" || (chapter === "benchmarks" && !tableHere);
  host.classList.toggle("show", on);
  bench.classList.toggle("show", tableHere);
  if (tableHere) {
    const r = plotRect();
    const row = labelRow(r.w) - 8;
    Object.assign(bench.style, { left: `${r.x}px`, top: `${r.y - row}px`, width: `${r.w}px`, height: `${r.h + row}px` });
  }
  if (!on || !M) return;
  const r = plotRect(), row = labelRow(r.w) - 8, top = r.y - row;
  const key = `${r.x},${top},${r.w},${r.h}`;
  if (key === figureKey) return;
  figureKey = key;
  Object.assign(host.style, { left: `${r.x}px`, top: `${top}px`, width: `${r.w}px`, height: `${r.h + row}px` });
  document.querySelectorAll(".tip").forEach((t) => t.remove());
  domainScatter(host, M.domains, M.domain_fit, M.combos.filter((c) => c.cka != null));
}

// Paper Sec. 4.2 / Fig. 4a as a table: mean and standard deviation of FOSCTTM over seeds, lower is better.
function benchTable(host, rows) {
  const name = (m) => ({ ours: "Ours", "platonic brain": "Platonic Brain" })[m] || m;
  const task = { NQ: "text encoder to text encoder", PBMC: "single-cell RNA to ATAC", "NSD fMRI": "brain to brain, across subjects" };
  // plain decimals, as many per benchmark as its smallest mean needs for two significant digits (at least four)
  const decimals = (members) => Math.max(4, Math.min(6, 1 - Math.floor(Math.log10(Math.min(...members.map((r) => r.foscttm))))));
  let html = '<table class="bench-table"><thead><tr><th scope="col">Benchmark</th><th scope="col">Method</th><th scope="col" class="num">FOSCTTM</th></tr></thead>';
  for (const g of [...new Set(rows.map((r) => r.benchmark))]) {
    const members = rows.filter((r) => r.benchmark === g);
    const best = Math.min(...members.map((r) => r.foscttm)), places = decimals(members), num = (v) => v.toFixed(places);
    html += "<tbody>";
    members.forEach((r, i) => {
      html += `<tr class="${r.method === "ours" ? "ours" : ""}">`;
      if (!i) html += `<th scope="rowgroup" rowspan="${members.length}"><span class="bench-name">${g}</span><span class="bench-task">${task[g] || ""}</span></th>`;
      html += `<td>${name(r.method)}</td><td class="num">${r.foscttm === best ? "<b>" : ""}${num(r.foscttm)}${r.foscttm === best ? "</b>" : ""} <span class="std">± ${num(r.std)}</span></td></tr>`;
    });
    html += "</tbody>";
  }
  host.innerHTML = html + "</table>";
}

// ------------------------------------------------------------------------------------------------ labels around the plot
function hud(metric) { $("metric").classList.toggle("on", metric); }
let lastLabel = "";
function setStageLabel(text) {
  const el = $("hud-stage");
  if (text !== lastLabel) { lastLabel = text; el.textContent = text; }
  el.classList.toggle("on", !!text);
}
let legendKey = "";
function legendHTML(kind) {
  if (!kind) return "";
  const shapes = `<span><svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.5" fill="currentColor"/></svg>image</span><span><svg viewBox="0 0 12 12"><path d="M6 1.5 11 10.5H1Z" fill="currentColor"/></svg>caption</span>`;
  const classes = kind === "super" || kind === "known" ? SUPER_NAMES.map((name, k) => `<span><i style="background:${rgb(SUPER[k])}"></i>${name}</span>`).join("") + `<span><i style="background:${rgb(SUPER[6], 0.5)}"></i>other</span>` : "";
  const known = kind === "known" ? `<span class="known-key"><svg class="wide" viewBox="0 0 30 12"><path d="M6 6H23" stroke="currentColor" stroke-width="1.5"/><circle cx="6" cy="6" r="3.6" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M24 1.6 28.6 10H19.4Z" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>known pair</span>` : "";
  return shapes + classes + known;
}
function legend(kind) {
  const el = $("legend");
  const key = kind || "";
  if (key !== legendKey) {
    legendKey = key;
    el.innerHTML = legendHTML(kind);
  }
  el.classList.toggle("on", !!kind);
}
// The row above the plot: the setting's name and the legend on the left, the FOSCTTM on the right.
function placeLabels() {
  const r = plotRect(), row = r.y - labelRow(r.w) + 4, root = document.documentElement.style;
  root.setProperty("--legend-y", `${row + (isMobile() ? 0 : STAGE_LINE)}px`);
  root.setProperty("--plot-x", `${r.x}px`);
  root.setProperty("--plot-y", `${r.y}px`);
  root.setProperty("--plot-w", `${r.w}px`);
  root.setProperty("--plot-h", `${r.h}px`);
  root.setProperty("--row-y", `${row}px`);
}

const callouts = new Map();
function showCallout(key, spec) {
  let el = callouts.get(key);
  if (!spec) { if (el) el.style.opacity = 0; return; }
  if (!el) { el = document.createElement("div"); el.className = "callout"; $("callouts").append(el); callouts.set(key, el); el.dataset.group = key.split("-")[0]; }
  if (el.dataset.html !== spec.html) { el.innerHTML = spec.html; el.dataset.html = spec.html; }
  el.style.left = `${spec.x}px`; el.style.top = `${spec.y}px`; el.style.opacity = 1;
}
function hideCallouts(group) { for (const [, el] of callouts) if (!group || el.dataset.group === group) el.style.opacity = 0; }

const hoverLeft = document.createElement("div"), hoverRight = document.createElement("div");
hoverLeft.className = hoverRight.className = "hc-col";
hoverLeft.style.position = hoverRight.style.position = "absolute";
let cardKey = "";
function renderCard(card, x, y, truth = null) {
  const key = JSON.stringify(card) + hover.slot;
  if (key !== cardKey) {
    cardKey = key;
    const badge = '<span class="badge">true match</span>';
    hoverLeft.innerHTML = card.left.map((c) => `${c.label ? `<span class="hc-lab">${c.label}</span>` : ""}<div class="hc-img-wrap"><img class="hc-img${c.small ? " small" : ""}" src="${c.img}" alt="${escapeHtml(c.alt)}"${c.match ? ' style="outline:2.5px solid var(--badge-bg);outline-offset:3px"' : ""}>${c.match ? badge : ""}</div>`).join("");
    hoverLeft.style.display = card.left.length ? "flex" : "none";
    hoverLeft.style.flexDirection = "column";
    hoverRight.innerHTML = card.right.map((c) => `<div class="hc-cap${c.main ? " main" : ""}">${c.label ? `<small>${c.label}${c.match ? badge : ""}</small>` : ""}${escapeHtml(c.text)}</div>`).join("");
    hoverRight.style.display = card.right.length ? "flex" : "none";
  }
  // the cards stay in the plot's side of the screen, clear of the panel
  const r = plotRect(), gap = 64, H = innerHeight;
  const lw = hoverLeft.offsetWidth, lh = hoverLeft.offsetHeight, rw = hoverRight.offsetWidth, rh = hoverRight.offsetHeight;
  const lx = Math.max(r.x - 12, x - gap - lw), rx = Math.min(x + gap, r.x + r.w + 12 - rw);
  const place = (h) => (!truth || Math.abs(truth[1] - y) < 12 ? y - h / 2 : truth[1] > y ? y + 24 - h : y - 24);
  hoverLeft.style.left = `${lx}px`; hoverLeft.style.top = `${clamp(place(lh), 8, H - lh - 8)}px`;
  hoverRight.style.left = `${rx}px`; hoverRight.style.top = `${clamp(place(rh), 8, H - rh - 8)}px`;
}
function clearCard() {
  hoverLeft.style.display = hoverRight.style.display = "none"; cardKey = "";
  $("inspector").hidden = true; document.body.classList.remove("inspecting");
}

// Phones: a tapped point is shown in the sheet, as a place is in a map app: what it is, then its neighbours.
function renderInspector(card, category) {
  const key = JSON.stringify(card) + hover.slot + category;
  if (key === cardKey) return;
  cardKey = key;
  const image = hover.slot < n;
  const color = rgb(SUPER[category] || C.neutral);
  const icon = image ? `<svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="5" fill="${color}"/></svg>` : `<svg viewBox="0 0 12 12"><path d="M6 1 11.5 11H.5Z" fill="${color}"/></svg>`;
  const head = `<div class="ins-head">${icon}<span>${image ? "Image" : "Caption"}</span>${category !== undefined && category < 6 ? `<span class="cat">${SUPER_NAMES[category]}</span>` : ""}</div>`;
  const badge = '<span class="badge">true match</span>';
  let body;
  if (image) {
    const list = card.right.length ? `<div class="ins-list">${card.right.map((c) => `<div class="item"><small>${c.label}${c.match ? badge : ""}</small>${escapeHtml(c.text)}</div>`).join("")}</div>` : "";
    body = `<div class="ins-image${list ? "" : " alone"}"><img src="${card.left[0].img}" alt="${escapeHtml(card.left[0].alt)}">${list}</div>`;
  } else {
    const grid = card.left.length ? `<div class="ins-grid">${card.left.map((c) => `<figure class="${c.match ? "match" : ""}"><small>${c.label.replace("nearest image", "nearest")}${c.match ? badge : ""}</small><img src="${c.img}" alt="${escapeHtml(c.alt)}"></figure>`).join("")}</div>` : "";
    body = `<p class="ins-caption">${escapeHtml(card.right[0].text)}</p>${grid}`;
  }
  const note = card.truth !== undefined ? `<p class="ins-note">${pct(card.closer)}% of the ${card.closerNoun} are closer than the true match.</p>` : "";
  $("inspector").querySelector(".ins-body").innerHTML = head + body + note;
  $("inspector").hidden = false;
  document.body.classList.add("inspecting");
  $("panel-body").scrollTop = 0;
  if (sheet.state === "peek") sheet.set("half");
}

// ------------------------------------------------------------------------------------------------ main loop
let frameRequested = false;
function requestFrame() { if (!frameRequested) { frameRequested = true; requestAnimationFrame(tick); } }

const out = { frame: null };
function tick(now) {
  frameRequested = false;
  if (!stage || !scenes.benchmarks) return;  // the sheet can ask for frames before the scenes exist
  const scene = scenes[SCENE_OF[chapter]];
  if (scene !== activeScene) { activeScene = scene; hover.slot = -1; clearCard(); hideCallouts(); }
  let keep = stepCamera(now) || stepAutoplay(now);
  placeLabels();
  const pr = plotRect();
  pointScale = isMobile() ? 1 : clamp(Math.min(pr.w, pr.h) / 440, 1, 1.5);
  if (chapter === "domains" || chapter === "benchmarks") sizeFigure();  // follows the panel as it slides in
  if (scene) {
    keep = (scene.compute(out.frame, now) || false) || keep;
    // the move into a chapter: from the frame that was on screen to the new chapter's own frame
    if (SNAP.t < 1) {
      const dt = Math.min(64, now - SNAP.last); SNAP.last = now;
      SNAP.t = reducedMotion ? 1 : clamp(SNAP.t + dt / 950);
      blend(out.frame, SNAP.frame, out.frame, ease(SNAP.t));
      keep = true;
    }
    stage.pos.set(out.frame.pos); stage.color.set(out.frame.col); stage.shape.set(out.frame.shape);
    // zoomed in, the points grow a little (much less than the distances between them)
    const grow = chapter === "title" ? 1 : Math.min(2, cam.z ** 0.35);
    if (grow !== 1) for (let i = 0; i < N; i++) stage.size[i] = out.frame.size[i] * grow; else stage.size.set(out.frame.size);
    stage.clipTop = chapter === "title" ? 0 : $("topbar").offsetHeight;
    stage.visible = !(["domains", "benchmarks"].includes(chapter) && SNAP.t >= 1);
    stage.drawOverlay = (ctx) => {
      // place the cards first, so that the scene's labels can keep clear of them
      const card = hover.slot >= 0 ? scene.card?.(hover.slot) : null;
      if (card && !isMobile()) renderCard(card, out.frame.pos[2 * hover.slot], out.frame.pos[2 * hover.slot + 1], card.truth !== undefined ? [out.frame.pos[2 * card.truth], out.frame.pos[2 * card.truth + 1]] : null);
      scene.overlay?.(ctx, now);
      if (chapter !== "method") { hideCallouts("pin"); hideCallouts("link"); }
      if (chapter !== "t2i") hideCallouts("t2i");
      if (chapter !== "pairs") hideCallouts("known");
      if (card && isMobile()) renderInspector(card, scene.categoryOf?.(hover.slot));
      else if (card) {
        const x = out.frame.pos[2 * hover.slot], y = out.frame.pos[2 * hover.slot + 1];
        ctx.strokeStyle = rgb(C.ink, 0.4); ctx.lineWidth = 1;
        for (const col of [hoverLeft, hoverRight]) {
          if (col.style.display === "none") continue;
          const r = col.getBoundingClientRect();
          const ex = col === hoverLeft ? r.right : r.left;
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, clamp(y, r.top + 8, r.bottom - 8)); ctx.stroke();
        }
      } else if (cardKey) clearCard();
    };
    placePlotHit(scene);
  }
  $("fit-view").hidden = !camMoved() || chapter === "title";
  stage.dirty = true;
  stage.render();
  if (keep || now < animatingUntil) requestFrame();
}

// A transparent layer over the plot takes the pointer: hover, tap, drag to move, wheel and pinch to zoom.
function placePlotHit(scene) {
  const hit = $("plot-hit");
  const on = chapter !== "title" && !!scene.hoverable?.();
  hit.classList.toggle("on", on);
  if (!on) return;
  const r = plotRect();
  const key = `${r.x},${r.y},${r.w},${r.h}`;
  if (hit.dataset.key !== key) { hit.dataset.key = key; Object.assign(hit.style, { left: `${r.x - 12}px`, top: `${r.y - 12}px`, width: `${r.w + 24}px`, height: `${r.h + 24}px` }); }
}

function nearestPoint(x, y, radius) {
  const fr = out.frame;
  let best = -1, bestD = radius * radius;
  for (let i = 0; i < N; i++) {
    if (fr.col[4 * i + 3] < 0.12) continue;
    const dx = fr.pos[2 * i] - x, dy = fr.pos[2 * i + 1] - y, d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best >= 0 && activeScene.card?.(best) ? best : -1;
}
function select(slot) { if (slot !== hover.slot) { hover.slot = slot; if (slot < 0) clearCard(); requestFrame(); } }

function gestures() {
  const hit = $("plot-hit"), pts = new Map();
  let pan = null, pinch = null;
  hit.addEventListener("pointerdown", (e) => {
    hit.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 1) pan = { x: e.clientX, y: e.clientY, moved: false, type: e.pointerType };
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), mx: (a[0] + b[0]) / 2, my: (a[1] + b[1]) / 2 };
      pan = null;
    }
  });
  hit.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) {  // a mouse without a button: hover
      if (e.pointerType !== "mouse" || !activeScene?.hoverable?.()) return;
      if (activeScene.hoverKnown?.(activeScene.pickKnown(e.clientX, e.clientY, 10)) >= 0) { select(-1); return; }
      select(nearestPoint(e.clientX, e.clientY, 13));
      return;
    }
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pinch && pts.size >= 2) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]), mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
      zoomAt(d / pinch.d, mx, my);
      cam.x += mx - pinch.mx; cam.y += my - pinch.my;
      Object.assign(pinch, { d, mx, my });
      requestFrame();
      return;
    }
    if (!pan) return;
    const dx = e.clientX - pan.x, dy = e.clientY - pan.y;
    if (!pan.moved && Math.hypot(dx, dy) < 6) return;
    pan.moved = true;
    cam.x += dx; cam.y += dy; camGoal.until = 0;
    pan.x = e.clientX; pan.y = e.clientY;
    hit.classList.add("grabbing");
    requestFrame();
  });
  const up = (e) => {
    pts.delete(e.pointerId);
    if (pts.size < 2) pinch = null;
    if (pan && !pan.moved && e.type === "pointerup") {
      // a tap or a click: inspect the nearest point, or let go of it; on phones a tap on empty space lowers the sheet
      const radius = pan.type === "touch" ? 26 : 13;
      if (activeScene.hoverKnown?.(activeScene.pickKnown(e.clientX, e.clientY, radius * 0.8)) >= 0) { select(-1); return; }
      const slot = nearestPoint(e.clientX, e.clientY, radius);
      select(slot);
      if (slot < 0 && isMobile() && sheet.state !== "peek") sheet.set("peek");
    }
    if (!pts.size) { pan = null; hit.classList.remove("grabbing"); }
  };
  hit.addEventListener("pointerup", up);
  hit.addEventListener("pointercancel", up);
  hit.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse" && !pts.size) { select(-1); activeScene?.hoverKnown?.(-1); } });
  hit.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018)), e.clientX, e.clientY);
    requestFrame();
  }, { passive: false });
  hit.addEventListener("dblclick", () => resetCamera());
  $("fit-view").addEventListener("click", () => resetCamera());
}

// ------------------------------------------------------------------------------------------------ boot
function decodeLabels(text) { const out = new Uint8Array(text.length); for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) - 48; return out; }

async function boot() {
  carve(); dialogs();
  readColors();
  const applyTheme = () => {
    readColors(); legendKey = "x"; figureKey = "";
    document.querySelector('meta[name="theme-color"]').content = getComputedStyle(document.documentElement).getPropertyValue("--stone").trim();
    sizeFigure();
    requestFrame();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
  document.querySelectorAll(".theme-btn").forEach((b) => b.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch {}
    applyTheme();
  }));
  $("callouts").append(hoverLeft, hoverRight);

  sheet = new Sheet($("panel"), {
    peek: () => {
      const body = $("panel-body"), grab = document.querySelector("#panel .grab").offsetHeight;
      const ref = document.body.classList.contains("inspecting") ? $("inspector").querySelector(".ins-head")
        : document.querySelector("article.active .tools") || document.querySelector("article.active .ch-head");
      return grab + (ref ? ref.getBoundingClientRect().bottom - body.getBoundingClientRect().top + body.scrollTop : 60) + 6;
    },
    onMove: () => { figureKey = ""; requestFrame(); sizeFigure(); },
  });
  sheet.enable(isMobile());
  clearCard();

  M = await loadManifest();
  n = M.n_train; N = 2 * n;
  if (M.n_val !== n) throw new Error("train and validation subsets must have the same size");
  document.querySelectorAll("[data-fill='stage.readout']").forEach((e) => { e.textContent = fmt(M.stage_foscttm.readout); });
  document.querySelectorAll("[data-fill='stage.refined']").forEach((e) => { e.textContent = fmt(M.stage_foscttm.refined); });
  superTrain = decodeLabels(M.super.train_image + M.super.train_text);
  superVal = decodeLabels(M.super.val);
  superT2I = decodeLabels(M.super.t2i);

  stage = new Stage($("stage"), $("overlay"), N);
  stage.resize();
  out.frame = makeFrame(N);
  SNAP.frame = makeFrame(N);
  D = await loadMethod(M);
  invImage = new Uint16Array(n); invText = new Uint16Array(n);
  for (let k = 0; k < n; k++) { invImage[D.toValImage[k]] = k; invText[D.toValText[k]] = n + k; }
  BOX = boxOf(D.unalignedImage, D.unalignedText, D.readout, D.refined, D.text);
  const pad = 0.04 * (BOX.x1 - BOX.x0);
  BOX = { x0: BOX.x0 - pad, x1: BOX.x1 + pad, y0: BOX.y0 - pad, y1: BOX.y1 + pad };

  TRUTH_SAMPLE = Array.from({ length: 80 }, (_, k) => Math.floor((k * n) / 80));
  scenes.method = method();
  scenes.overview = overview();
  scenes.hero = hero();
  scenes.results = results();
  scenes.pairs = pairs();
  scenes.geometry = geometry();
  scenes.domains = chartScene("domains");
  scenes.benchmarks = chartScene("benchmarks");

  // chapters, steps, play and replay
  document.addEventListener("click", (e) => {
    const go = e.target.closest("[data-go]");
    if (go) { setChapter(go.dataset.go); return; }
    const step = e.target.closest(".seg [data-step]");
    if (step) { setStep(+step.dataset.step); return; }
    if (e.target.closest("[data-play]")) { playMethod(); return; }
    if (e.target.closest("[data-replay]")) { replay(); return; }
  });
  $("enter").addEventListener("click", () => setChapter("overview"));
  $("home").addEventListener("click", () => { carveTitle($("hero").querySelector(".title")); setChapter("title"); });
  $("inspector").querySelector(".close").addEventListener("click", () => { hover.slot = -1; clearCard(); requestFrame(); });
  // on phones, a swipe up on the title also opens the atlas
  let swipe = null;
  addEventListener("pointerdown", (e) => { swipe = chapter === "title" && e.pointerType === "touch" && !e.target.closest("#stone-wrap, a, button, dialog") ? [e.clientY, performance.now()] : null; });
  addEventListener("pointerup", (e) => { if (swipe && swipe[0] - e.clientY > 60 && performance.now() - swipe[1] < 700) setChapter("overview"); swipe = null; });
  gestures();

  // the address names the chapter (and the method's step), so that reloading and shared links keep their place
  const fromHash = () => { const [id, step] = location.hash.slice(1).split("-"); return [CHAPTERS.includes(id) ? id : "title", +step || 0]; };
  const [id0, step0] = fromHash();
  if (id0 !== "title") scenes.hero.skip();
  else carveTitle($("hero").querySelector(".title"));
  setChapter(id0, step0, { push: false, instant: true, morph: false });
  addEventListener("popstate", () => { const [id, step] = fromHash(); setChapter(id, step, { push: false }); });

  // Everything past the title loads when the browser is idle.
  const later = async () => {
    [captionsTrain, altTrain, captionsVal] = await Promise.all(["data/captions_train.json", "data/alt_train.json", "data/captions_val.json"].map(loadJSON));
    await scenes.results.ready();
    T2I = await loadT2I(M);
    captionsT2I = await loadJSON("data/captions_t2i.json");
    await scenes.pairs.ready();
    scenes.geometry.ready();
    requestFrame();
  };
  (window.requestIdleCallback || ((f) => setTimeout(f, 300)))(later);

  // Keys: left and right move through the method's steps and the chapters; Escape lets go of a point.
  addEventListener("keydown", (e) => {
    if (e.target.closest?.("input, select, textarea, dialog") || e.altKey || e.ctrlKey || e.metaKey) return;
    const i = CHAPTERS.indexOf(chapter);
    if (e.key === "ArrowRight") {
      e.preventDefault();
      if (chapter === "method" && scenes.method.manual < 3) setStep(scenes.method.manual + 1);
      else if (i < CHAPTERS.length - 1) setChapter(CHAPTERS[i + 1]);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (chapter === "method" && scenes.method.manual > 0) setStep(scenes.method.manual - 1);
      else if (i > 0) setChapter(CHAPTERS[i - 1], CHAPTERS[i - 1] === "method" ? 3 : 0);
    } else if (e.key === "Escape") { hover.slot = -1; activeScene?.hoverKnown?.(-1); clearCard(); requestFrame(); }
    else if (e.key === "0" || e.key === "f") resetCamera();
  });

  let pending = 0;
  const relayout = () => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => {
      sheet.enable(isMobile());
      stage.resize(); figureKey = ""; sizeFigure();
      hover.slot = -1; clearCard(); requestFrame();
    });
  };
  addEventListener("resize", relayout);
  addEventListener("orientationchange", relayout);
  window.visualViewport?.addEventListener("resize", relayout);
  sheetQuery.addEventListener("change", relayout);
  stage.onRestore = requestFrame;
  document.fonts?.ready.then(() => { legendMeasure.key = ""; figureKey = ""; sizeFigure(); if (isMobile()) sheet.set(sheet.state, false); requestFrame(); });
  requestFrame();
  window.__hover = hover;  // for automated tests
  window.__chapter = () => chapter;
  window.__plotRect = () => plotRect();
  window.__slotPos = (slot) => [out.frame.pos[2 * slot], out.frame.pos[2 * slot + 1]];
  window.__cam = cam;
  window.__scene = () => activeScene;
}

boot();
