// Schematic viewer for the ASO:mRNA duplex.
//
// The coordinates come from the server's idealized A-form PDB, so nothing here
// predicts structure; the drawing exists to show length, register, gapmer
// architecture and where the oligo sits along the transcript. Residues are
// painted back to front with depth shading, which is what gives the helix its
// sense of volume on a flat canvas.

const COLORS = {
  wing: "#6ed9b2",
  core: "#60a8fa",
  rna: "#f7ab64",
  flank: "#8d8f7f",
  pair: "#cfe3e6",
};
// The stage colour, used to fade distant atoms into the background.
const BACKGROUND = [12, 47, 52];

const ZOOM_MIN = 0.45;
const ZOOM_MAX = 6;

// HELM monomers look like d(A)[sp] or [moe]([m5C])[sp]: sugar, base, linkage.
const MONOMER = /(\[[^\]]+\]|[a-z])\((\[[^\]]+\]|[A-Z])\)(\[[^\]]+\])?/g;

const SUGARS = { "[moe]": "2′-MOE", m: "2′-OMe", d: "DNA", r: "RNA" };
const LINKAGES = { "[sp]": "phosphorothioate", "": "3′ terminus" };

function parseHelm(helm) {
  if (typeof helm !== "string") return [];
  const inner = helm.slice(helm.indexOf("{") + 1, helm.lastIndexOf("}"));
  const units = [];
  for (const match of inner.matchAll(MONOMER)) {
    const [, sugar, base, linkage = ""] = match;
    units.push({
      sugar: SUGARS[sugar] || sugar.replace(/[[\]]/g, ""),
      base: base.replace(/[[\]]/g, ""),
      modified: base.startsWith("["),
      linkage: LINKAGES[linkage] ?? linkage.replace(/[[\]]/g, ""),
      thio: linkage === "[sp]",
    });
  }
  return units;
}

// Mix a hex colour toward the background. `nearness` of 1 keeps it, 0 buries it.
function shade(hex, nearness) {
  const n = Math.max(0, Math.min(1, nearness));
  const value = parseInt(hex.slice(1), 16);
  const channels = [value >> 16 & 255, value >> 8 & 255, value & 255];
  const mixed = channels.map((c, i) => Math.round(BACKGROUND[i] + (c - BACKGROUND[i]) * n));
  return `rgb(${mixed.join(",")})`;
}

export class DuplexView {
  constructor(canvas, { onHover } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onHover = onHover;
    this.strands = [];
    this.units = [];
    this.picks = [];
    this.reset();
    new ResizeObserver(() => this.draw()).observe(canvas);

    let drag;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      drag = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 1 };
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag) {
        this.hover(event);
        return;
      }
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (drag.pan) {
        this.panX += dx;
        this.panY += dy;
      } else {
        this.angle += dx * 0.012;
        this.tilt += dy * 0.012;
      }
      drag.x = event.clientX;
      drag.y = event.clientY;
      this.draw();
    });
    canvas.addEventListener("lostpointercapture", () => { drag = undefined; });
    canvas.addEventListener("pointerleave", () => this.setHovered(undefined));

    canvas.addEventListener("wheel", (event) => {
      if (!this.strands.length) return;
      event.preventDefault();
      // Trackpad pinch arrives as a ctrl-wheel; both gestures zoom.
      this.zoomBy(Math.exp(-event.deltaY * 0.0016));
    }, { passive: false });

    canvas.addEventListener("keydown", (event) => {
      const rotation = {
        ArrowLeft: [-0.2, 0], ArrowRight: [0.2, 0],
        ArrowUp: [0, -0.2], ArrowDown: [0, 0.2],
      }[event.key];
      if (rotation) {
        event.preventDefault();
        this.angle += rotation[0];
        this.tilt += rotation[1];
        this.draw();
        return;
      }
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        this.zoomBy(1.18);
      } else if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        this.zoomBy(1 / 1.18);
      } else if (event.key === "0") {
        event.preventDefault();
        this.reset();
      }
    });
  }

  zoomBy(factor) {
    this.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, this.zoom * factor));
    this.draw();
  }

  reset() {
    this.angle = 0.48;
    this.tilt = -0.14;
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.draw();
  }

  clear() {
    this.strands = [];
    this.units = [];
    this.picks = [];
    this.setHovered(undefined);
    this.draw();
  }

  /**
   * @param pdb       PDB text from /api/structure
   * @param sequence  the ASO sequence, 5'->3'
   * @param design    { wing5, wing3 } from the chemistry template
   * @param context   { helm, flank5, flank3 } for annotation and mRNA extent
   */
  load(pdb, sequence, { wing5, wing3 }, { helm = "", flank5 = "", flank3 = "" } = {}) {
    if (!this.context) throw new Error("Canvas unavailable. Download the PyMOL script to view.");
    const length = sequence.length;
    this.units = parseHelm(helm);
    const a = Array.from({ length }, (_, index) => {
      const unit = this.units[index];
      const wing = index < wing5 || index >= length - wing3;
      return {
        color: wing ? COLORS.wing : COLORS.core,
        base: sequence[index],
        label: `ASO ${index + 1} · ${unit?.base || sequence[index]}`,
        detail: unit
          ? `${unit.sugar}${unit.modified ? ` · ${unit.base}` : ""} · ${unit.linkage}`
          : wing ? "wing" : "DNA core",
        thio: unit?.thio ?? index < length - 1,
      };
    });

    // Chain B is the transcript: flank5 + binding window + flank3, 5'->3'.
    const rnaLength = flank5.length + length + flank3.length;
    const b = Array.from({ length: rnaLength }, (_, index) => {
      const paired = index >= flank5.length && index < flank5.length + length;
      return {
        color: paired ? COLORS.rna : COLORS.flank,
        paired,
        label: `mRNA ${index + 1}`,
        detail: paired ? "paired with the ASO" : "unpaired transcript",
      };
    });

    for (const line of pdb.split("\n")) {
      if (!line.startsWith("ATOM  ") || !["A", "B"].includes(line[21])) continue;
      const atom = line.slice(12, 16).trim();
      if (atom !== "P" && atom !== "N1") continue;
      const residue = (line[21] === "A" ? a : b)[Number(line.slice(22, 26)) - 1];
      if (residue) residue[atom] = [30, 38, 46].map((s) => Number(line.slice(s, s + 8)));
    }
    if ([...a, ...b].some(({ P, N1 }) => !P || !N1 || ![...P, ...N1].every(Number.isFinite))) {
      throw new Error("The duplex file is incomplete.");
    }

    this.strands = [a, b];
    this.sequence = sequence;
    this.flank5 = flank5.length;
    // Frame on the bound window rather than the whole transcript: the oligo is
    // the subject, and the flanks reading off both edges is the point of them.
    const bound = a.map(({ P }) => P[2]);
    const all = [...a, ...b].map(({ P }) => P[2]);
    this.center = (Math.min(...bound) + Math.max(...bound)) / 2;
    const duplex = Math.max(...bound) - Math.min(...bound);
    const whole = Math.max(...all) - Math.min(...all);
    this.span = duplex + 0.5 * (whole - duplex);
    this.reset();
  }

  setHovered(pick) {
    const changed = this.hovered?.key !== pick?.key;
    this.hovered = pick;
    if (changed) {
      this.canvas.style.cursor = pick ? "pointer" : "";
      this.onHover?.(pick);
      this.draw();
    }
  }

  hover(event) {
    if (!this.picks.length) return this.setHovered(undefined);
    const box = this.canvas.getBoundingClientRect();
    const x = event.clientX - box.left;
    const y = event.clientY - box.top;
    let best;
    for (const pick of this.picks) {
      const distance = Math.hypot(pick.x - x, pick.y - y);
      // Prefer the nearest to the pointer, breaking ties toward the viewer.
      if (distance <= pick.r + 5 && (!best || distance < best.distance)) {
        best = { ...pick, distance };
      }
    }
    this.setHovered(best);
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
    this.picks = [];
    if (!this.strands.length) return;

    const scale = this.zoom * Math.min(w / (this.span + 34), h / 48);
    const ca = Math.cos(this.angle), sa = Math.sin(this.angle);
    const ct = Math.cos(this.tilt), st = Math.sin(this.tilt);
    const project = ([x, y, z]) => {
      const horizontal = z - this.center;
      const vertical = x * sa + y * ca;
      return [
        w / 2 + this.panX + scale * (horizontal * ct - vertical * st),
        h / 2 + this.panY + scale * (horizontal * st + vertical * ct),
        x * ca - y * sa,
      ];
    };

    const [a, b] = this.strands.map((strand, chain) => strand.map((residue, index) => ({
      ...residue,
      key: `${chain}:${index}`,
      p: project(residue.P),
      n: project(residue.N1),
    })));

    // Depth runs over the helix radius, so normalise against the widest atom.
    const depths = [...a, ...b].map(({ p }) => p[2]);
    const near = Math.max(...depths), far = Math.min(...depths);
    const nearness = (z) => (near === far ? 1 : 0.34 + 0.66 * ((z - far) / (near - far)));

    const layers = [];
    const paint = (z, fn) => layers.push([z, fn]);
    const stroke = (from, to, color, width, alpha) => {
      const z = (from[2] + to[2]) / 2;
      paint(z, () => {
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = shade(color, nearness(z));
        ctx.lineWidth = Math.max(0.6, width * scale * 0.11);
        ctx.beginPath();
        ctx.moveTo(from[0], from[1]);
        ctx.lineTo(to[0], to[1]);
        ctx.stroke();
      });
    };

    // Base pairs, drawn only across the bound window. G:C gets the heavier rung.
    a.forEach((residue, i) => {
      const partner = b[this.flank5 + this.sequence.length - 1 - i];
      if (!partner) return;
      const gc = "GC".includes(this.sequence[i]);
      stroke(residue.n, partner.n, COLORS.pair, gc ? 1.5 : 0.95, gc ? 0.66 : 0.4);
    });

    for (const strand of [a, b]) {
      strand.forEach((residue, i) => {
        const { p, n, color } = residue;
        stroke(p, n, color, residue.paired === false ? 1.1 : 1.5,
          residue.paired === false ? 0.32 : 0.6);
        if (i) {
          const previous = strand[i - 1];
          stroke(previous.p, p, color, 3.1, 0.95);
          // Phosphorothioate linkages carry a marker at the backbone midpoint.
          if (previous.thio && scale * 0.24 >= 1.3) {
            const z = (previous.p[2] + p[2]) / 2;
            paint(z + 0.01, () => {
              ctx.globalAlpha = 0.9;
              ctx.fillStyle = shade("#ffd9a8", nearness(z));
              const mx = (previous.p[0] + p[0]) / 2;
              const my = (previous.p[1] + p[1]) / 2;
              const r = Math.max(1.5, scale * 0.24);
              ctx.beginPath();
              ctx.moveTo(mx, my - r);
              ctx.lineTo(mx + r, my);
              ctx.lineTo(mx, my + r);
              ctx.lineTo(mx - r, my);
              ctx.closePath();
              ctx.fill();
            });
          }
        }
        const radius = Math.max(1.6, scale * 0.52 * (0.72 + 0.4 * nearness(p[2])));
        this.picks.push({ ...residue, x: p[0], y: p[1], r: radius });
        const lit = this.hovered?.key === residue.key;
        paint(p[2], () => {
          ctx.globalAlpha = 1;
          ctx.fillStyle = shade(color, nearness(p[2]));
          ctx.beginPath();
          ctx.arc(p[0], p[1], radius, 0, 2 * Math.PI);
          ctx.fill();
          if (lit) {
            ctx.globalAlpha = 1;
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        });
      });
    }

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    layers.sort((left, right) => left[0] - right[0]).forEach(([, fn]) => fn());
    ctx.globalAlpha = 1;
  }
}
