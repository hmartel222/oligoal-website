import "./molecule-view.js";

const $ = (id) => document.getElementById(id);
const viewer = $("design-molecule");
let runVersion = 0;
let currentRun;
let exportRequest;
let results;
let resultParams;
let researchModels = [];

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

function showExportStatus(message) {
  $("export-status").textContent = message;
  $("export-status").hidden = !message;
}

const present = (value) => Number.isFinite(value);
const formatted = (value) => present(value) ? value.toFixed(2) : "—";
const site = (row) => `${row.start + 1}–${row.end}`;

async function run(event) {
  event.preventDefault();
  currentRun?.abort();
  exportRequest?.abort();
  currentRun = new AbortController();
  const version = ++runVersion;
  const runParams = Object.fromEntries(new FormData($("design-form")));
  if (!runParams.chemistry) {
    status("Choose an available design profile.", true);
    return;
  }
  results = undefined;
  resultParams = undefined;
  showExportStatus("");
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
    resultParams = runParams;
    render();
    status("");
  } catch (error) {
    if (version === runVersion && error.name !== "AbortError") status(error.message, true);
  } finally {
    if (version === runVersion) $("run").removeAttribute("aria-busy");
  }
}

function render(preserveSelection = false) {
  const body = $("results-body");
  const previous = preserveSelection && body.querySelector("tr.active")?.rowIndex - 1;
  const previousChecks = preserveSelection && new Set(
    [...body.querySelectorAll(".bench-check:checked")].map((checkbox) => checkbox.value)
  );
  const { transcript, summary, rows, endpoints } = results;
  const columns = Object.entries(endpoints).filter(([, spec]) => spec.status === "scored");
  const methods = {
    ppo: "Exploratory PPO",
    design_space_search: "Design-space search",
    atlas_efficacy: "Atlas 20-mer efficacy",
    immune_panel: "2′-OMe immune panel",
  };
  $("provenance").textContent = `${transcript.gene} · ${transcript.transcript_id} · ${transcript.length.toLocaleString()} nt`;
  const assay = summary.assay;
  const context = assay
    ? ` · ${assay.cell_line}, ${assay.transfection_method.toLowerCase()}, ${assay.dosage_nm.toLocaleString()} nM, ${assay.treatment_period_hrs} h`
    : "";
  $("method").textContent = `${methods[summary.method]} · ${summary.scored.toLocaleString()} scored · ${summary.ranked_by}${context}`;
  $("coverage").textContent = rows.length
    ? `Showing ${rows.length} of ${summary.ranked_count.toLocaleString()} ranked candidates. Coordinates are 1-based on this transcript.`
    : "No candidates qualified for this run.";
  $("research-form").hidden = !researchModels.length || results.chemistry !== "ps_moe_20_gapmer";
  $("research-status").textContent = results.localModel || "";
  $("download").disabled = !rows.length;
  $("download-full").disabled = !rows.length;
  $("send-bench").hidden = !results.bench_run;
  $("send-bench").disabled = !rows.length || (preserveSelection && !previousChecks.size);
  const headings = node("tr");
  if (results.bench_run) headings.append(node("th", "Send"));
  headings.append(...["Rank", "ASO 5′–3′", "Site"].map((label) => node("th", label)));
  for (const [, spec] of columns) {
    const header = node("th", spec.title);
    header.append(node("span", spec.units));
    headings.append(header);
  }
  if (results.localScores) headings.append(node("th", "Local % inhibition"));
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
    if (results.bench_run) {
      const cell = node("td");
      const checked = node("input");
      checked.type = "checkbox";
      checked.className = "bench-check";
      checked.value = row.candidate_id;
      checked.checked = !previousChecks || previousChecks.has(row.candidate_id);
      checked.setAttribute("aria-label", `Send candidate ${index + 1} to Bench`);
      cell.addEventListener("click", (event) => event.stopPropagation());
      checked.addEventListener("change", () => {
        $("send-bench").disabled = !$("results-body").querySelector(".bench-check:checked");
      });
      cell.append(checked);
      tr.append(cell);
    }
    tr.append(node("td", String(index + 1)), sequence, node("td", site(row)));
    tr.append(...columns.map(([key]) => node("td", formatted(row.predictions[key]))));
    if (results.localScores) tr.append(node("td", formatted(results.localScores[row.candidate_id])));
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
  if (rows.length) select(Number.isInteger(previous) && previous >= 0 ? previous : 0);
}

async function select(index) {
  const profile = results;
  const row = profile.rows[index];
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
  if (results.localScores) {
    const definition = node("div");
    definition.className = "endpoint";
    definition.append(node("dt", results.localModel), node("dd", `${formatted(results.localScores[row.candidate_id])} % inhibition`));
    values.append(definition);
  }
  $("predictions").replaceChildren(values);
  await viewer.show(row, profile.transcript);
}

$("design-form").addEventListener("submit", run);

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = node("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

$("download").addEventListener("click", () => {
  if (!results?.csv) return;
  download(new Blob([results.csv], { type: "text/csv;charset=utf-8" }), "aso-candidates.csv");
});

$("download-full").addEventListener("click", async () => {
  if (!resultParams || $("download-full").disabled) return;
  const request = exportRequest = new AbortController();
  const version = runVersion;
  $("download-full").disabled = true;
  showExportStatus("Preparing full CSV…");
  try {
    const response = await fetch("/api/panel.csv", {
      method: "POST",
      signal: request.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...resultParams, scope: "full" }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "The full export failed.");
    }
    const csv = await response.blob();
    if (version !== runVersion) return;
    download(csv, `${results.transcript.gene}-${results.chemistry}-full.csv`);
    showExportStatus("");
  } catch (error) {
    if (version === runVersion && error.name !== "AbortError") showExportStatus(error.message);
  } finally {
    if (version === runVersion) $("download-full").disabled = false;
  }
});

$("send-bench").addEventListener("click", () => {
  const candidates = [...$("results-body").querySelectorAll(".bench-check:checked")]
    .map((checkbox) => checkbox.value);
  if (!results?.bench_run || !candidates.length) return;
  try {
    sessionStorage.setItem("oligoal-bench-handoff", JSON.stringify({ run: results.bench_run, candidates }));
    window.location.assign("/bench");
  } catch {
    showExportStatus("Enable browser session storage to send candidates to Bench.");
  }
});

$("research-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const run = results;
  const control = event.submitter;
  const model = $("research-model").value;
  control.disabled = true;
  $("research-status").textContent = "Running checkpoint locally…";
  try {
    const response = await fetch("/api/research/score", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({run: run.bench_run, model}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    if (results !== run) return;
    run.localScores = data.scores;
    run.localModel = researchModels.find((entry) => entry.id === model).label;
    render(true);
  } catch (error) {
    if (results === run) $("research-status").textContent = error.message;
  } finally {
    control.disabled = false;
  }
});

async function loadProfiles() {
  try {
    const response = await fetch("/api/meta", { cache: "no-store" });
    if (!response.ok) return;
    const meta = await response.json();
    $("atlas-profile").disabled = !meta.atlas_efficacy_configured;
    $("open-bench").hidden = !meta.bench_configured;
    researchModels = meta.research_models || [];
    $("research-model").replaceChildren(...researchModels.map(({id, label}) => {
      const option = node("option", label);
      option.value = id;
      return option;
    }));
    if (meta.research_default) $("research-model").value = meta.research_default;
    if (results) render();
  } catch {
    // The default profiles remain available while metadata is unreachable.
  }
}

loadProfiles();
