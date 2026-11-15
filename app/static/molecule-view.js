import {DuplexView} from "./duplex.js";

class MoleculeView extends HTMLElement {
  connectedCallback() {
    if (this.canvas) return;
    this.innerHTML = `
      <div class="molecule-head">
        <h3>Molecular viewer</h3>
        <div class="molecule-tools">
          <button type="button" data-action="binding" aria-pressed="true">Binding site</button>
          <button type="button" data-action="transcript" aria-pressed="false">Full transcript</button>
          <select aria-label="Atomic representation"><option value="sticks">Ball and stick</option><option value="space">Space-filling</option></select>
          <button type="button" data-action="zoomIn" aria-label="Zoom in">+</button>
          <button type="button" data-action="zoomOut" aria-label="Zoom out">−</button>
          <button type="button" data-action="reset" aria-label="Reset view">Reset</button>
          <button type="button" data-action="expand" aria-label="Expand molecular viewer">Expand</button>
        </div>
      </div>
      <div class="structure-stage">
        <canvas tabindex="0" aria-label="ASO–mRNA model. Arrow keys rotate, plus and minus zoom. Select an atom for details."></canvas>
        <p class="viewer-status" role="status" hidden></p>
        <div class="molecule-overlay">
          <output class="atom-readout" aria-live="polite"></output>
          <div class="legend">
            <span class="wing">ASO wings</span><span class="core">ASO core</span><span class="rna">mRNA</span>
          </div>
          <label><input type="checkbox" class="motion-control"> Motion preview</label>
        </div>
      </div>
      <div class="transcript-track">
        <button type="button" data-action="track" title="Focus on the ASO binding site">
          <span class="track-highlight"></span>
        </button>
        <span class="transcript-caption"></span>
      </div>
      <p class="structure-note">Drag to rotate, scroll to zoom, and right-drag to pan. Select an atom to inspect it. Template-built duplex with a schematic mRNA trace and illustrative motion.</p>
      <div class="exports">
        <button type="button" class="secondary" data-action="pml">Download PyMOL script</button>
        <button type="button" class="secondary" data-action="pdb">Download duplex PDB</button>
      </div>`;
    this.canvas = this.querySelector("canvas");
    if (this.id === "design-molecule") this.canvas.id = "viewer";
    this.readout = this.querySelector(".atom-readout");
    this.status = this.querySelector(".viewer-status");
    this.motion = this.querySelector(".motion-control");
    this.motion.checked = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.addEventListener("click", (event) => {
      const action = event.target.closest("[data-action]")?.dataset.action;
      if (!action) return;
      if (action === "expand") {
        this.toggleExpanded();
      } else if (["pml", "pdb"].includes(action)) {
        this.download(action);
      } else if (["binding", "transcript", "track"].includes(action)) {
        this.focusMode(action === "track" ? "binding" : action);
      } else if (action === "reset") {
        this.view?.reset();
      } else {
        this.view?.zoom(action === "zoomIn" ? .77 : 1.30);
      }
    });
    this.querySelector("select").addEventListener("change", (event) => this.view?.setStyle(event.target.value));
    this.motion.addEventListener("change", () => this.view?.setMotion(this.motion.checked));
    this.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.classList.contains("viewer-expanded")) this.toggleExpanded();
    });
  }

  focusMode(mode) {
    this.querySelector('[data-action="binding"]').setAttribute("aria-pressed", mode === "binding");
    this.querySelector('[data-action="transcript"]').setAttribute("aria-pressed", mode === "transcript");
    this.view?.setMode(mode);
  }

  toggleExpanded() {
    const expanded = this.classList.toggle("viewer-expanded");
    document.body.classList.toggle("molecule-expanded", expanded);
    const control = this.querySelector('[data-action="expand"]');
    control.textContent = expanded ? "Close" : "Expand";
    control.setAttribute("aria-label", expanded ? "Close full-screen viewer" : "Expand molecular viewer");
    if (expanded) this.canvas.focus();
    else control.focus();
  }

  setStatus(message) {
    this.status.hidden = !message;
    this.status.textContent = message;
  }

  clear() {
    this.request?.abort();
    this.candidate = null;
    this.view?.clear();
  }

  async show(candidate, transcript) {
    this.clear();
    this.candidate = candidate;
    this.transcript = transcript;
    this.readout.textContent = "";
    this.focusMode("binding");
    const count = transcript.length;
    const track = this.querySelector(".track-highlight");
    track.style.left = 100 * candidate.start / count + "%";
    track.style.width = 100 * (candidate.end - candidate.start) / count + "%";
    this.querySelector(".transcript-caption").textContent =
      `5′ · ${count.toLocaleString()} nt · 3′   |   ASO binds nt ${(candidate.start + 1).toLocaleString()}–${candidate.end.toLocaleString()}`;
    this.setStatus("Loading molecular model…");
    const request = this.request = new AbortController();
    try {
      const pdb = await this.fetchStructure("pdb", request.signal);
      if (request !== this.request || request.signal.aborted) return;
      if (!this.view) {
        try {
          this.view = new DuplexView(this.canvas, (atom) => {
            const residue = atom.chain === "B" ? this.candidate.start + atom.residue : atom.residue;
            const strand = atom.chain === "B" ? "mRNA" : "ASO";
            this.readout.textContent = `${strand} nt ${residue} · ${atom.kind} · ${atom.name} (${atom.element}) · ${atom.xyz.toArray().map((p) => p.toFixed(2)).join(", ")} Å`;
          });
          this.view.setMotion(this.motion.checked);
        } catch {
          throw new Error("WebGL is unavailable. Download the model to open it in PyMOL.");
        }
      }
      this.view.load(pdb, candidate, count);
      this.focusMode(this.querySelector('[data-action="transcript"]').getAttribute("aria-pressed") === "true" ? "transcript" : "binding");
      this.setStatus("");
    } catch (error) {
      if (error.name !== "AbortError" && request === this.request) this.setStatus(error.message);
    }
  }

  async fetchStructure(format, signal) {
    const response = await fetch("/api/structure", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      signal,
      body: JSON.stringify({
        sequence: this.candidate.sequence, chemistry: this.candidate.chemistry,
        target: this.candidate.target_window, format,
      }),
    });
    if (!response.ok) throw new Error("Could not load the atomic structure.");
    return response.text();
  }

  async download(format) {
    if (!this.candidate) return;
    const gene = this.transcript.gene.replace(/[^a-z0-9._-]/gi, "_");
    const name = `${gene}-${this.candidate.start + 1}.${format}`;
    try {
      const source = await this.fetchStructure(format);
      const url = URL.createObjectURL(new Blob([source], {type: "text/plain"}));
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
    } catch (error) {
      this.setStatus(error.message);
    }
  }
}

customElements.define("molecule-view", MoleculeView);
