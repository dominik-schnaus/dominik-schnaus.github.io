// Small page-wide touches: carved headings with a puff of dust, a few drifting marks, dialogs, copying.

export const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// Split a heading into letters (kept inside words so that lines still wrap between words).
function split(element) {
  let i = 0;
  const walk = (node) => {
    for (const child of [...node.childNodes]) {
      if (child.nodeType === Node.TEXT_NODE) {
        const fragment = document.createDocumentFragment();
        for (const part of child.textContent.split(/(\s+)/)) {
          if (!part) continue;
          if (/^\s+$/.test(part)) { fragment.append(part); continue; }
          const word = document.createElement("span");
          word.style.whiteSpace = "nowrap";
          word.style.display = "inline-block";
          for (const ch of part) {
            const span = document.createElement("span");
            span.className = "ch";
            span.style.setProperty("--i", i++);
            span.textContent = ch;
            word.append(span);
          }
          fragment.append(word);
        }
        child.replaceWith(fragment);
      } else if (child.nodeType === Node.ELEMENT_NODE && child.tagName !== "BR") {
        walk(child);
      }
    }
  };
  walk(element);
  element.setAttribute("aria-label", element.textContent.replace(/\s+/g, " ").trim());
  return i;
}

const DUST = ["var(--d-green)", "var(--d-purple)", "var(--d-blue)", "var(--d-yellow)"];
// One puff of particles at (x, y), in the MCML colours: a few that fly off and fade.
export function puff(x, y, count = 3) {
  for (let k = 0; k < count; k++) {
    const p = document.createElement("i");
    p.className = "dust";
    const t = 0.7 + Math.random() * 0.7;
    p.style.cssText = `left:${x + (Math.random() - 0.5) * 6}px;top:${y + (Math.random() - 0.5) * 6}px;--s:${(1.2 + Math.random() * 2.8).toFixed(1)}px;` +
      `--c:${DUST[k % DUST.length]};--t:${t.toFixed(2)}s;--dx:${(Math.random() * 46 - 12).toFixed(0)}px;` +
      `--dy:${(18 + Math.random() * 46).toFixed(0)}px;--r:${(Math.random() * 360).toFixed(0)}deg`;
    document.body.append(p);
    setTimeout(() => p.remove(), t * 1000 + 50);
  }
}

function dust(element) {
  const letters = element.querySelectorAll(".ch");
  letters.forEach((letter, k) => {
    if (k % 2) return;
    setTimeout(() => { const r = letter.getBoundingClientRect(); puff(r.right, r.bottom - r.height * (0.2 + Math.random() * 0.5), 2); }, k * 26 + 60);
  });
}

// Takeaways: the text is cut in from left to right, line by line, while dust falls from the chisel.
export function carveBlock(element) {
  if (element.classList.contains("carved")) return;
  element.classList.add("carved");
  if (reducedMotion) return;
  const r = element.getBoundingClientRect();
  const lines = Math.max(1, Math.round(r.height / 22));
  for (let k = 0; k < 22; k++) {
    const u = k / 21;
    setTimeout(() => {
      const rr = element.getBoundingClientRect();
      const line = Math.min(lines - 1, Math.floor(Math.random() * lines));
      puff(rr.left + u * rr.width, rr.top + (line + 0.8) * (rr.height / lines), 2);
    }, u * 1000);
  }
}

// Titles are split into letters once; each is carved (with a puff of stone dust) the first time its slide shows.
export function carve() {
  for (const heading of document.querySelectorAll(".carve")) split(heading);
  if (reducedMotion) document.querySelectorAll(".carve").forEach((h) => h.classList.add("carved"));
}
export function carveTitle(heading) {
  if (heading.classList.contains("carved")) return;
  heading.classList.add("carved");
  if (!reducedMotion) requestAnimationFrame(() => dust(heading));
}

export function dialogs() {
  const copyTarget = document.getElementById("bibtex-dialog");
  if (copyTarget) copyTarget.textContent = document.getElementById("bibtex-text").textContent;
  document.addEventListener("click", (event) => {
    const opener = event.target.closest("[data-open]");
    if (opener) {
      const dialog = document.getElementById(`dlg-${opener.dataset.open}`);
      if (dialog) { dialog.showModal(); event.preventDefault(); }
      return;
    }
    const closer = event.target.closest("[data-close]");
    if (closer) { closer.closest("dialog").close(); return; }
    if (event.target.tagName === "DIALOG") event.target.close();  // a click on the backdrop
    const copy = event.target.closest("[data-copy]");
    if (copy) {
      navigator.clipboard?.writeText(document.getElementById(copy.dataset.copy).textContent).then(() => {
        const label = copy.textContent;
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = label; }, 1400);
      });
    }
  });
}
