import { DuplexView } from "./duplex.js";

const $ = (id) => document.getElementById(id);
const viewer = new DuplexView($("viewer"));
let runVersion = 0;
let currentRun;
let structureRequest;
let results;

function node(tag, text) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  return element;
}

function status(message, error = false) {
  $("status").hidden = !message;
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function showViewerStatus(message) {
  $("viewer-status").textContent = message;
  $("viewer-status").hidden = !message;
}

const present = (value) => Number.isFinite(value);
const formatted = (value) => present(value) ? value.toFixed(2) : "—";
const site = (row) => `${row.start + 1}–${row.end}`;

async function run(event) {
  event.preventDefault();
  currentRun?.abort();
  structureRequest?.abort();
  currentRun = new AbortController();
  const version = ++runVersion;
  const runParams = Object.fromEntries(new FormData($("design-form")));
  results = undefined;
  $("workspace").hidden = true;
  $("detail").hidden = true;
  $("run").setAttribute("aria-busy", "true");
  status(`Running design for ${runParams.gene.trim().toUpperCase() || "pasted transcript"}…`);
  try {
    const response = await fetch("/api/design", {
      method: "POST",
      signal: currentRun.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runParams),
    });
    const next = await response.json();
    if (version !== runVersion) return;
    if (!response.ok) throw new Error(next.error || "The design run failed.");
    results = next;
    render();
    status("");
  } catch (error) {
    if (version === runVersion && error.name !== "AbortError") status(error.message, true);
  } finally {
    if (version === runVersion) $("run").removeAttribute("aria-busy");
  }
}

function render() {
  const { transcript, summary, rows, endpoints } = results;
  const columns = Object.entries(endpoints).filter(([, spec]) => spec.status === "scored");
  const methods = {
    ppo: "Exploratory PPO",
    design_space_search: "Design-space search",
    immune_panel: "2′-OMe immune panel",
  };
  $("provenance").textContent = `${transcript.gene} · ${transcript.transcript_id} · ${transcript.length.toLocaleString()} nt`;
  $("method").textContent = `${methods[summary.method]} · ${summary.scored} scored · ${summary.ranked_by}`;
  $("coverage").textContent = rows.length
    ? `Showing ${rows.length} candidates. Coordinates are 1-based on this transcript.`
    : "No candidates qualified for this run.";
  $("download").disabled = !rows.length;
  const headings = node("tr");
  headings.append(...["Rank", "ASO 5′–3′", "Site"].map((label) => node("th", label)));
  for (const [, spec] of columns) {
    const header = node("th", spec.title);
    header.append(node("span", spec.units));
    headings.append(header);
  }
  $("results-head").replaceChildren(headings);
  const fragment = document.createDocumentFragment();
  rows.forEach((row, index) => {
    const tr = node("tr");
    const button = node("button", row.sequence);
    button.type = "button";
    button.className = "sequence";
    button.setAttribute("aria-label", `Inspect candidate ${index + 1}, site ${site(row)}`);
    button.setAttribute("aria-pressed", "false");
    const sequence = node("td");
    sequence.append(button);
    tr.append(node("td", String(index + 1)), sequence, node("td", site(row)));
    tr.append(...columns.map(([key]) => node("td", formatted(row.predictions[key]))));
    tr.addEventListener("click", () => {
      select(index);
      if (window.matchMedia("(max-width: 780px)").matches) {
        $("detail").scrollIntoView({ block: "start" });
      }
    });
    fragment.append(tr);
  });
  $("results-body").replaceChildren(fragment);
  $("workspace").hidden = false;
  if (rows.length) select(0);
}

async function select(index) {
  const profile = results;
  const row = profile.rows[index];
  const version = runVersion;
  structureRequest?.abort();
  const request = structureRequest = new AbortController();
  $("detail").hidden = false;
  for (const [i, tr] of Array.from($("results-body").rows).entries()) {
    tr.classList.toggle("active", i === index);
    tr.querySelector("button").setAttribute("aria-pressed", String(i === index));
  }
  $("detail-title").textContent = `Candidate ${index + 1}`;
  $("detail-site").textContent = `${profile.transcript.gene} · nt ${site(row)}`;
  const rna = row.target_window.replaceAll("T", "U");
  $("alignment").textContent =
    `ASO    5′  ${row.sequence}  3′\n           ${"|".repeat(rna.length)}\nmRNA   3′  ${Array.from(rna).reverse().join("")}  5′`;
  $("helm").textContent = row.helm;
  const values = document.createDocumentFragment();
  for (const [key, spec] of Object.entries(profile.endpoints)) {
    const definition = node("div");
    definition.className = "endpoint";
    const value = row.predictions[key];
    const text = present(value) ? `${formatted(value)} ${spec.units}`
      : spec.status === "not_applicable" ? "Outside chemistry scope" : "Model unavailable";
    const reading = node("dd", text);
    if (present(value) && spec.model) {
      reading.append(node("small", spec.model));
      if (key === "neurotox_moe" && (value < 0 || value > 7)) {
        reading.append(node("small", "Outside measured FOB range (0–7)"));
      }
    }
    definition.append(node("dt", spec.title), reading);
    values.append(definition);
  }
  $("predictions").replaceChildren(values);
  const params = new URLSearchParams({ sequence: row.sequence, target: rna, chemistry: row.chemistry });
  const filename = `${profile.transcript.gene.replace(/[^a-z0-9._-]/gi, "_")}-${row.start + 1}`;
  $("pdb-link").href = `/api/structure?${params}&download=1`;
  $("pdb-link").download = `${filename}.pdb`;
  $("pml-link").href = `/api/structure?${params}&format=pml`;
  $("pml-link").download = `${filename}.pml`;
  viewer.clear();
  showViewerStatus("Loading duplex…");
  try {
    const response = await fetch(`/api/structure?${params}`, { signal: request.signal });
    if (!response.ok) throw new Error("Could not load the duplex.");
    const pdb = await response.text();
    if (version !== runVersion || request !== structureRequest) return;
    viewer.load(pdb, row.sequence, profile.design);
    showViewerStatus("");
  } catch (error) {
    if (error.name !== "AbortError" && version === runVersion && request === structureRequest) {
      showViewerStatus(error.message);
    }
  }
}

$("design-form").addEventListener("submit", run);
$("reset-view").addEventListener("click", () => viewer.reset());
$("download").addEventListener("click", () => {
  if (!results?.csv) return;
  const url = URL.createObjectURL(new Blob([results.csv], { type: "text/csv;charset=utf-8" }));
  const link = node("a");
  link.href = url;
  link.download = "aso-candidates.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
