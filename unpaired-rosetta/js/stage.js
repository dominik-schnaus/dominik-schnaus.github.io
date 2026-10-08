// One canvas for every point cloud of the page, plus a 2-D overlay canvas for lines and labels.
// The scenes compute screen positions on the CPU (a few thousand points: cheaper than keeping shaders in sync)
// and the stage only uploads and draws them, and only when something changed.
// WebGL draws the points when it is available; otherwise (blocked GPU, emulators, a context the browser took away)
// the same points are drawn with Canvas 2D, which is slower but plenty for two thousand points.

const VERT = `
attribute vec2 a_pos; attribute vec4 a_color; attribute float a_size; attribute float a_shape;
uniform vec2 u_view; uniform float u_dpr;
varying vec4 v_color; varying float v_shape; varying float v_size;
void main() {
  vec2 clip = a_pos / u_view * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  gl_PointSize = a_size * u_dpr;
  v_color = a_color; v_shape = a_shape; v_size = a_size * u_dpr;
}`;

const FRAG = `
precision mediump float;
varying vec4 v_color; varying float v_shape; varying float v_size;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float aa = 2.0 / max(v_size, 1.0);
  float alpha;
  if (v_shape < 0.5) {            // circle: an image
    alpha = 1.0 - smoothstep(1.0 - aa, 1.0, length(p) * 1.12);
  } else {                        // triangle: a caption
    vec2 q = vec2(p.x, -p.y + 0.18);
    float d = max(abs(q.x) * 0.866 + q.y * 0.5, -q.y) - 0.5;
    alpha = 1.0 - smoothstep(-aa, aa, d);
  }
  float a = v_color.a * alpha;
  if (a < 0.004) discard;
  gl_FragColor = vec4(v_color.rgb * a, a);
}`;

export const SHAPE = { image: 0, text: 1 };

export class Stage {
  constructor(canvas, overlay, capacity) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.ctx = overlay.getContext("2d");
    // Note whether a frame drew anything, so that an empty overlay is hidden rather than left for the compositor.
    for (const name of ["stroke", "fill", "fillText", "strokeText", "drawImage"]) {
      const original = this.ctx[name].bind(this.ctx);
      this.ctx[name] = (...args) => { this.used = true; return original(...args); };
    }
    this.n = capacity;
    this.pos = new Float32Array(capacity * 2);
    this.color = new Float32Array(capacity * 4);
    this.size = new Float32Array(capacity);
    this.shape = new Float32Array(capacity);
    this.dirty = true;
    this.visible = true;
    this.drawOverlay = null;
    this.clipTop = 0;  // nothing is drawn above this line (the top bar), so a zoomed cloud passes under it
    this.gl = canvas.getContext("webgl", { antialias: false, alpha: true, premultipliedAlpha: true, powerPreference: "low-power" });
    if (this.gl) {
      this.initGL();
      // Phones take WebGL contexts away (background tabs, memory pressure); draw nothing until it is back.
      canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this.lost = true; });
      canvas.addEventListener("webglcontextrestored", () => { this.lost = false; this.initGL(); this.dirty = true; this.onRestore?.(); });
    } else {
      this.points2d = canvas.getContext("2d");  // fallback renderer
    }
  }

  initGL() {
    const gl = this.gl;
    const program = this.program(VERT, FRAG);
    gl.useProgram(program);
    this.u = { view: gl.getUniformLocation(program, "u_view"), dpr: gl.getUniformLocation(program, "u_dpr") };
    this.buffers = {};
    for (const [name, size] of [["a_pos", 2], ["a_color", 4], ["a_size", 1], ["a_shape", 1]]) {
      const buffer = gl.createBuffer();
      const location = gl.getAttribLocation(program, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.n * size * 4, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
      this.buffers[name] = buffer;
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  program(vs, fs) {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(program);
    return program;
  }

  // The canvases cover exactly the visible viewport, in CSS pixels set here, so that what is drawn in window
  // coordinates lines up with the page (100vh can be taller than the visible area on phones).
  resize() {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.width = document.documentElement.clientWidth || window.innerWidth;
    this.height = window.innerHeight;
    for (const c of [this.canvas, this.overlay]) {
      c.style.width = `${this.width}px`;
      c.style.height = `${this.height}px`;
      c.width = Math.round(this.width * this.dpr);
      c.height = Math.round(this.height * this.dpr);
    }
    this.dirty = true;
  }

  render() {
    if (!this.dirty) return;
    this.dirty = false;
    const gl = this.gl;
    if (gl && !this.lost) {
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (this.visible) {
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(0, 0, this.canvas.width, Math.max(0, Math.round((this.height - this.clipTop) * this.dpr)));
        const upload = (name, data) => { gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[name]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, data); };
        upload("a_pos", this.pos); upload("a_color", this.color); upload("a_size", this.size); upload("a_shape", this.shape);
        gl.uniform2f(this.u.view, this.width, this.height);
        gl.uniform1f(this.u.dpr, this.dpr);
        gl.drawArrays(gl.POINTS, 0, this.n);
      }
    } else if (this.points2d) {
      this.render2d();
    }
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    this.used = false;
    if (this.visible && this.drawOverlay) {
      ctx.save();
      ctx.beginPath(); ctx.rect(0, this.clipTop, this.width, this.height); ctx.clip();
      this.drawOverlay(ctx);
      ctx.restore();
    }
    const shown = this.used ? "visible" : "hidden";
    if (this.overlay.style.visibility !== shown) this.overlay.style.visibility = shown;
  }

  // The same circles and triangles as the shader, with Canvas 2D.
  render2d() {
    const ctx = this.points2d;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    if (!this.visible) return;
    ctx.save();
    ctx.beginPath(); ctx.rect(0, this.clipTop, this.width, this.height); ctx.clip();
    const { pos, color, size, shape } = this;
    for (let i = 0; i < this.n; i++) {
      const a = color[4 * i + 3];
      if (a < 0.004) continue;
      const x = pos[2 * i], y = pos[2 * i + 1];
      if (x < -20 || y < -20 || x > this.width + 20 || y > this.height + 20) continue;
      ctx.fillStyle = `rgba(${(color[4 * i] * 255) | 0},${(color[4 * i + 1] * 255) | 0},${(color[4 * i + 2] * 255) | 0},${a})`;
      const r = size[i] / 2;
      ctx.beginPath();
      if (shape[i] < 0.5) ctx.arc(x, y, r / 1.12, 0, 6.2832);
      else { ctx.moveTo(x, y - r * 0.82); ctx.lineTo(x + r * 0.87, y + r * 0.68); ctx.lineTo(x - r * 0.87, y + r * 0.68); ctx.closePath(); }
      ctx.fill();
    }
    ctx.restore();
  }
}
