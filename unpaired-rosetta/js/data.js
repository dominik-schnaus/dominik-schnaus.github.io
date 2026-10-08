// Loading and decoding of the binary files written by prep/export.py (little-endian throughout).

const cache = new Map();

export function fetchOnce(url, kind = "arrayBuffer") {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: ${r.status}`);
      return r[kind]();
    }));
  }
  return cache.get(url);
}

class Reader {
  constructor(buffer) { this.buffer = buffer; this.offset = 0; }
  take(Type, count) {
    const bytes = count * Type.BYTES_PER_ELEMENT;
    const out = new Type(this.buffer.slice(this.offset, this.offset + bytes));
    this.offset += bytes;
    return out;
  }
}

// int16 positions -> Float32 layout units
function positions(int16, scale) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) out[i] = int16[i] / scale;
  return out;
}

export async function loadManifest() {
  return fetchOnce("data/manifest.json", "json");
}

export async function loadMethod(m) {
  const r = new Reader(await fetchOnce("data/method.bin"));
  const n = m.n_train, s = m.scale;
  return {
    unalignedImage: positions(r.take(Int16Array, 2 * n), s),
    unalignedText: positions(r.take(Int16Array, 2 * n), s),
    readout: positions(r.take(Int16Array, 2 * n), s),
    refined: positions(r.take(Int16Array, 2 * n), s),
    text: positions(r.take(Int16Array, 2 * n), s),
    clusterImage: r.take(Uint8Array, n),
    clusterText: r.take(Uint8Array, n),
    nnImage: r.take(Uint16Array, 3 * n),
    nnText: r.take(Uint16Array, 3 * n),
    readoutNnImage: r.take(Uint16Array, 3 * n),
    readoutNnText: r.take(Uint16Array, 3 * n),
    tints: r.take(Uint8Array, 3 * n),
    toValImage: r.take(Uint16Array, n),
    toValText: r.take(Uint16Array, n),
  };
}

// One validation layout: positions, neighbours and closer fractions of images 0..n-1, then captions n..2n-1.
export async function loadValBlock(url, m) {
  const r = new Reader(await fetchOnce(url));
  const n = m.n_val;
  const pos = positions(r.take(Int16Array, 4 * n), m.scale);
  const nn = r.take(Uint16Array, 6 * n);
  const closerRaw = r.take(Uint16Array, 2 * n);
  const closer = new Float32Array(2 * n);
  for (let i = 0; i < 2 * n; i++) closer[i] = closerRaw[i] / 65535;
  return { pos, nn, closer };
}

export async function loadT2I(m) {
  const r = new Reader(await fetchOnce("data/t2i.bin"));
  const k = m.n_t2i, c = m.t2i.captions.length, p = m.t2i.pairs.length, s = m.scale;
  return {
    vision: positions(r.take(Int16Array, 2 * k), s),
    language: positions(r.take(Int16Array, 2 * k), s),
    captions: positions(r.take(Int16Array, 2 * c), s),
    mapped: positions(r.take(Int16Array, 2 * c * p), s),  // [pairs][caption][2]
    nn: r.take(Uint16Array, 3 * c * p),                     // [pairs][caption][3]
  };
}

export async function loadJSON(url) { return fetchOnce(url, "json"); }
