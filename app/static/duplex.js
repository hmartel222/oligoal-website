const COLORS = { wing: "#6ed9b2", core: "#60a8fa", rna: "#f7ab64" };

export class DuplexView {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.strands = [];
    this.reset();
    new ResizeObserver(() => this.draw()).observe(canvas);
    let drag;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      drag = [event.clientX, event.clientY];
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      this.angle += (event.clientX - drag[0]) * .012;
      this.tilt += (event.clientY - drag[1]) * .012;
      drag = [event.clientX, event.clientY];
      this.draw();
    });
    canvas.addEventListener("lostpointercapture", () => { drag = undefined; });
    canvas.addEventListener("keydown", (event) => {
      const rotation = { ArrowLeft: [-.2, 0], ArrowRight: [.2, 0],
        ArrowUp: [0, -.2], ArrowDown: [0, .2] }[event.key];
      if (!rotation) return;
      event.preventDefault();
      this.angle += rotation[0];
      this.tilt += rotation[1];
      this.draw();
    });
  }

  reset() {
    this.angle = .48;
    this.tilt = -.22;
    this.draw();
  }

  clear() {
    this.strands = [];
    this.draw();
  }

  load(pdb, sequence, { wing5, wing3 }) {
    if (!this.context) throw new Error("Canvas unavailable. Download the PyMOL script to view.");
    const length = sequence.length;
    const a = Array.from({ length }, (_, index) => ({
      color: index < wing5 || index >= length - wing3 ? COLORS.wing : COLORS.core,
    }));
    const b = Array.from({ length }, () => ({ color: COLORS.rna }));
    for (const line of pdb.split("\n")) {
      if (!line.startsWith("ATOM  ") || !["A", "B"].includes(line[21])) continue;
      const atom = line.slice(12, 16).trim();
      if (atom !== "P" && atom !== "N1") continue;
      const residue = (line[21] === "A" ? a : b)[Number(line.slice(22, 26)) - 1];
      if (residue) residue[atom] = [30, 38, 46].map((start) => Number(line.slice(start, start + 8)));
    }
    if ([...a, ...b].some(({ P, N1 }) => !P || !N1 || ![...P, ...N1].every(Number.isFinite))) {
      throw new Error("The duplex file is incomplete.");
    }
    this.strands = [a, b];
    this.sequence = sequence;
    this.center = (a[0].P[2] + a.at(-1).P[2]) / 2;
    this.reset();
  }

  draw() {
    const { canvas, context: ctx } = this;
    const { clientWidth: w, clientHeight: h } = canvas;
    if (!ctx || !w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.strands.length) return;
    const scale = Math.min(w / (2 * this.center + 27), h / 47);
    const ca = Math.cos(this.angle), sa = Math.sin(this.angle);
    const ct = Math.cos(this.tilt), st = Math.sin(this.tilt);
    const project = ([x, y, z]) => {
      const horizontal = z - this.center;
      const vertical = x * sa + y * ca;
      return [w / 2 + scale * (horizontal * ct - vertical * st),
        h / 2 + scale * (horizontal * st + vertical * ct), x * ca - y * sa];
    };
    const [a, b] = this.strands.map((strand) => strand.map(({ P, N1, color }) => ({
      p: project(P), base: project(N1), color,
    })));
    const layers = [];
    const line = (from, to, stroke, width, alpha) => {
      layers.push([(from[2] + to[2]) / 2, () => {
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(from[0], from[1]);
        ctx.lineTo(to[0], to[1]);
        ctx.stroke();
      }]);
    };
    a.forEach((residue, i) => {
      const gc = "GC".includes(this.sequence[i]);
      line(residue.base, b[a.length - i - 1].base, "#cfe3e6", gc ? 1.4 : .8, gc ? .62 : .36);
    });
    for (const strand of [a, b]) {
      strand.forEach(({ p, base, color }, i) => {
        line(p, base, color, 1.6, .68);
        if (i) line(strand[i - 1].p, p, color, 2.7, .92);
        layers.push([p[2], () => {
          ctx.globalAlpha = .96;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(p[0], p[1], Math.max(2, scale * .58), 0, 2 * Math.PI);
          ctx.fill();
        }]);
      });
    }
    ctx.lineCap = "round";
    layers.sort((left, right) => left[0] - right[0]).forEach(([, paint]) => paint());
    ctx.globalAlpha = 1;
  }
}
