// A bottom sheet for phones, as in map apps: it rests at three heights (peek, half, full), follows the finger
// while it is dragged by its grab areas, and is flung to the next height by a quick swipe. The cloud above it
// is laid out around its current height, which the stage reads every frame.

export class Sheet {
  constructor(el, { peek, onMove }) {
    this.el = el;
    this.peekOf = peek;          // () => height of the peek state (measured from the content)
    this.onMove = onMove;        // called whenever the height changes
    this.state = "half";
    this.enabled = false;
    this.drag = null;
    // the handle, the chapter chips and each chapter's title row move the sheet; the text scrolls normally
    for (const zone of el.querySelectorAll("[data-grab]")) this.grabZone(zone);
    // a drag that started on a chip must not also choose that chapter
    el.addEventListener("click", (e) => { if (this.suppress) { e.stopPropagation(); e.preventDefault(); this.suppress = false; } }, true);
  }
  heights() {
    const H = window.innerHeight;
    return { peek: Math.min(this.peekOf(), H * 0.4), half: Math.round(H * 0.46), full: Math.round(H - 56) };
  }
  height() { return this.enabled ? this.el.getBoundingClientRect().height : 0; }
  enable(on) {
    this.enabled = on;
    this.el.classList.toggle("sheet", on);
    if (on) this.set(this.state, false); else { this.el.style.height = ""; this.el.style.transition = ""; }
  }
  set(state, animate = true) {
    if (!this.enabled) return;
    this.state = state;
    this.el.dataset.state = state;
    this.el.style.transition = animate ? "height .32s cubic-bezier(.2, .8, .2, 1)" : "none";
    this.el.style.height = `${this.heights()[state]}px`;
    const t0 = performance.now();
    const follow = () => { this.onMove(); if (performance.now() - t0 < 380) requestAnimationFrame(follow); };
    requestAnimationFrame(follow);
  }
  cycle() { this.set({ peek: "half", half: "full", full: "peek" }[this.state]); }
  grabZone(zone) {
    zone.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button, a, input, select") && e.pointerType !== "touch") return;
      this.start(e);
    });
  }
  start(e) {
    if (!this.enabled || this.drag) return;
    const h0 = this.el.getBoundingClientRect().height;
    this.drag = { y0: e.clientY, h0, last: [[performance.now(), e.clientY]], moved: false, id: e.pointerId, target: e.target };
    const move = (ev) => {
      if (ev.pointerId !== this.drag.id) return;
      const dy = ev.clientY - this.drag.y0;
      if (!this.drag.moved && Math.abs(dy) < 6) return;
      if (!this.drag.moved) { this.drag.moved = true; this.suppress = true; this.el.style.transition = "none"; }
      const { peek, full } = this.heights();
      const h = Math.max(peek * 0.85, Math.min(full, this.drag.h0 - dy));
      this.el.style.height = `${h}px`;
      this.drag.last.push([performance.now(), ev.clientY]);
      if (this.drag.last.length > 5) this.drag.last.shift();
      this.onMove();
      ev.preventDefault();
    };
    const up = (ev) => {
      if (ev.pointerId !== this.drag?.id) return;
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      removeEventListener("pointercancel", up);
      const d = this.drag; this.drag = null;
      if (!d.moved) { this.suppress = false; if (!d.target.closest("button, a, input, select")) this.cycle(); return; }
      setTimeout(() => { this.suppress = false; }, 0);
      const [t0, y0] = d.last[0], [t1, y1] = d.last[d.last.length - 1];
      const v = (y1 - y0) / Math.max(1, t1 - t0);  // px per ms, positive is downward
      const h = this.el.getBoundingClientRect().height, hs = this.heights();
      const order = ["peek", "half", "full"];
      let state;
      if (Math.abs(v) > 0.45) {  // a fling goes one step in its direction from where the finger let go
        const above = order.filter((s) => hs[s] > h + 8), below = order.filter((s) => hs[s] < h - 8);
        state = v < 0 ? (above[0] || "full") : (below[below.length - 1] || "peek");
      } else state = order.reduce((a, b) => (Math.abs(hs[a] - h) < Math.abs(hs[b] - h) ? a : b));
      this.set(state);
    };
    addEventListener("pointermove", move, { passive: false });
    addEventListener("pointerup", up);
    addEventListener("pointercancel", up);
  }
}
