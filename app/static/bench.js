import "./molecule-view.js";

const $ = (id) => document.getElementById(id);
const HANDOFF = "oligoal-bench-handoff";
const profiles = {
  ps_moe_gapmer: "19-nt MOE",
  ps_moe_20_gapmer: "Atlas 20-nt efficacy",
  ps_2ome_gapmer: "20-nt OMe immune panel",
};
const types = {
  inhibition: ["Inhibition", "% inhibition"],
  dose_response: ["Dose-response curve", "% inhibition"],
  ic50: ["IC₅₀", "nM", "µM"],
  neurotoxicity: ["Neurotoxicity", "FOB score"],
  hepatotoxicity: ["Hepatotoxicity", "U/L", "IU/L", "mg/dL", "mmol/L", "µmol/L", "g/L", "ratio"],
};
const methods = {
  import: "Design shortlist",
  atlas_prior: "Atlas prior ranking",
  bayesian_atlas_v1: "Measurement-updated Atlas ranking",
  design_rank: "Design ranking",
  research_ppo: "Research PPO",
};
let state;
let selectedId;
let editing;
let resultRequest;
let loadVersion = 0;

function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(label, click, className = "secondary") {
  const node = element("button", label);
  node.type = "button";
  node.className = className;
  node.addEventListener("click", click);
  return node;
}

function message(text = "", error = false) {
  $("bench-status").textContent = text;
  $("bench-status").classList.toggle("error", error);
}

async function api(action, body) {
  const response = await fetch("/api/bench/" + action, body === undefined ? { cache: "no-store" } : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || "Workspace request failed.");
    error.status = response.status;
    throw error;
  }
  return data;
}

function route(project, batch) {
  const url = new URL("/bench", location.origin);
  if (project) url.searchParams.set("project", project);
  if (batch) url.searchParams.set("batch", batch);
  history.replaceState(null, "", url);
}

function activeBatch() {
  const key = new URLSearchParams(location.search).get("batch");
  return state.batches.find((batch) => batch.id === key) || state.batches[0];
}

function format(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

function showLogin() {
  $("bench").hidden = true;
  $("empty").hidden = true;
  $("logout").hidden = true;
  $("login").hidden = false;
}

async function load({keepSelection = false} = {}) {
  const version = ++loadVersion;
  const initial = new URLSearchParams(location.search);
  try {
    const query = initial.get("project") ? "?" + new URLSearchParams({project: initial.get("project")}) : "";
    const next = await api("state" + query);
    if (version !== loadVersion) return;
    state = next;
    $("login").hidden = true;
    $("logout").hidden = false;
    const handoff = sessionStorage.getItem(HANDOFF);
    if (handoff) {
      try {
        const saved = await api("import", JSON.parse(handoff));
        sessionStorage.removeItem(HANDOFF);
        route(saved.project, saved.batch);
        message("Candidates sent to Bench Workspace.");
        await load();
        return;
      } catch (error) {
        if ([400, 410].includes(error.status)) sessionStorage.removeItem(HANDOFF);
        message(error.message, true);
      }
    }
    $("empty").hidden = Boolean(state.project);
    $("bench").hidden = !state.project;
    if (state.project) render(keepSelection);
  } catch (error) {
    if (version !== loadVersion) return;
    if (error.status === 401) showLogin();
    else message(error.message, true);
  }
}

function render(keepSelection = false) {
  const project = state.project;
  const batch = activeBatch();
  const transcript = project.transcript;
  const assay = project.assay;
  route(project.id, batch.id);
  const options = state.projects.map((item) => {
    const option = element("option", `${item.transcript.gene} · ${profiles[item.chemistry]}`);
    option.value = item.id;
    return option;
  });
  $("projects").replaceChildren(...options);
  $("projects").value = project.id;
  $("batches").replaceChildren(...state.batches.map((item) => {
    const tab = button(item.name, () => {
      message();
      route(project.id, item.id);
      render();
    });
    tab.setAttribute("aria-current", String(item.id === batch.id));
    return tab;
  }));
  $("batch-title").textContent = batch.name;
  $("rename-form").hidden = true;
  const evidence = batch.evidence;
  const adapted = evidence.compatible_observations?.length || 0;
  $("batch-context").textContent = evidence.expert
    ? `Research PPO · ${evidence.expert} expert · ${adapted} measured candidates in policy state`
    : `${methods[evidence.method]} · ${batch.candidates.length} candidates · ${adapted} measurements used for adaptation`;
  $("project-context").textContent = [
    transcript.gene, transcript.transcript_id, profiles[project.chemistry],
    assay && `${assay.cell_line}, ${assay.transfection_method}, ${assay.dosage_nm.toLocaleString()} nM, ${assay.treatment_period_hrs} h`,
  ].filter(Boolean).join(" · ");
  $("budget-status").textContent = `${state.spent} of ${project.budget} assays recorded · ${state.pending.length} proposed`;
  $("budget").value = project.budget;
  $("notes").value = state.notes.body;
  const events = [
    ...state.batches.map((item) => ({time: item.created, text: `${item.name} created`})),
    ...state.measurements.map((item) => ({
      time: item.created, text: `${types[item.result.kind][0]} ${item.revision > 1 ? "corrected" : "recorded"}`,
    })),
  ].sort((a, b) => b.time.localeCompare(a.time));
  $("activity").replaceChildren(...events.slice(0, 8).map((entry) =>
    element("li", `${entry.text} · ${new Date(entry.time).toLocaleString()}`)));
  const endpoint = Object.keys(project.endpoints).find((key) => project.endpoints[key].status === "scored");
  $("score-heading").textContent = endpoint ? project.endpoints[endpoint].title : "Prediction";
  $("candidates").replaceChildren(...batch.candidates.map((candidate) => {
    const id = candidate.candidate_id;
    const count = state.measurements.filter((item) => item.candidate === id).length;
    const prediction = candidate.bench_prediction ?? candidate.predictions[endpoint];
    const tr = element("tr");
    const link = button(candidate.sequence, () => showCandidate(id), "sequence");
    const seq = element("td");
    seq.append(link);
    tr.dataset.id = id;
    tr.append(seq, element("td", `${candidate.start + 1}–${candidate.end}`),
      element("td", format(prediction)), element("td", String(count)),
      element("td", count ? "Result logged" : state.pending.includes(id) ? "Proposed" : "Available"));
    return tr;
  }));
  const requested = keepSelection && batch.candidates.some((row) => row.candidate_id === selectedId);
  showCandidate(requested ? selectedId : batch.candidates[0].candidate_id);
}

function showCandidate(id) {
  selectedId = id;
  editing = undefined;
  $("result-form").hidden = true;
  const row = activeBatch().candidates.find((candidate) => candidate.candidate_id === id);
  for (const tr of $("candidates").rows) tr.classList.toggle("active", tr.dataset.id === id);
  $("candidate-details").hidden = false;
  $("selected-sequence").textContent = row.sequence;
  $("selected-site").textContent = `Site ${row.start + 1}–${row.end} · design rank ${row.design_rank}`;
  $("helm").textContent = row.helm;
  if ($("candidate-structure").open) $("bench-molecule").show(row, state.project.transcript);
  const fields = Object.entries(state.project.endpoints)
    .filter(([key]) => Number.isFinite(row.predictions[key]))
    .map(([key, endpoint]) => [endpoint.title, `${format(row.predictions[key])} ${endpoint.units} · ${endpoint.model}`]);
  if (Number.isFinite(row.bench_prediction)) {
    fields.push(["Batch recommendation prediction", `${format(row.bench_prediction)} % inhibition`]);
  }
  $("predictions").replaceChildren(...fields.map(([name, value]) => {
    const field = element("div");
    field.className = "endpoint";
    field.append(element("dt", name), element("dd", value));
    return field;
  }));
  const records = state.measurements.filter((item) => item.candidate === id);
  $("results").replaceChildren(...records.map((item) => {
    const result = item.result;
    const marker = {lt: "<", gt: ">", exact: ""}[result.qualifier];
    const value = result.kind === "dose_response"
      ? result.points.map(([dose, pct]) => `${dose} nM → ${pct}%`).join(", ")
      : `${marker}${result.value} ${result.unit}`;
    const entry = element("li");
    const conditions = result.context;
    const description = [conditions.species, conditions.cell_line, conditions.method,
      conditions.route, conditions.dose_nm != null && `${conditions.dose_nm} nM`,
      conditions.hours != null && `${conditions.hours} h`, conditions.detail].filter(Boolean);
    entry.append(element("strong", `${types[result.kind][0]} · ${result.status === "failed" ? "Assay failed" : value}`),
      element("small", [...description, `revision ${item.revision}`].join(" · ")));
    if (result.notes) entry.append(element("div", result.notes));
    entry.append(button("Correct result", () => editResult(item)));
    return entry;
  }));
  if (!records.length) $("results").append(element("li", "No measurements yet."));
}

function syncEditor() {
  const kind = $("kind").value;
  const measured = $("result-status").value === "measured";
  const current = $("unit").value;
  $("unit").replaceChildren(...types[kind].slice(1).map((unit) => element("option", unit)));
  if (types[kind].slice(1).includes(current)) $("unit").value = current;
  $("qualifier-field").hidden = kind !== "ic50" || !measured;
  $("value-field").hidden = !measured || kind === "dose_response";
  $("value").required = !$("value-field").hidden;
  $("points-field").hidden = !measured || kind !== "dose_response";
  $("points").required = !$("points-field").hidden;
  $("result-notes").required = !measured;
}

function editResult(record) {
  const form = $("result-form");
  form.reset();
  editing = record;
  resultRequest = crypto.randomUUID();
  const result = record?.result;
  const assay = state.project.assay;
  const context = result?.context ?? {
    species: assay?.cell_line_species, cell_line: assay?.cell_line, method: assay?.transfection_method,
    dose_nm: assay?.dosage_nm, hours: assay?.treatment_period_hrs,
  };
  $("editor-title").textContent = record ? "Correct result" : "Log a result";
  $("kind").value = result?.kind ?? "inhibition";
  $("result-status").value = result?.status ?? "measured";
  syncEditor();
  if (result) $("unit").value = result.unit;
  $("qualifier").value = result?.qualifier ?? "exact";
  $("value").value = result?.value ?? "";
  $("points").value = (result?.points ?? []).map((point) => point.join(", ")).join("\n");
  $("species").value = context.species ?? "";
  $("cell-line").value = context.cell_line ?? "";
  $("method").value = context.method ?? "";
  $("route").value = context.route ?? "";
  $("dose").value = context.dose_nm ?? "";
  $("hours").value = context.hours ?? "";
  $("assay-detail").value = context.detail ?? "";
  $("result-notes").value = result?.notes ?? "";
  $("charge").checked = result?.charge ?? true;
  $("charge").disabled = Boolean(record);
  form.hidden = false;
  form.scrollIntoView({block: "nearest"});
  $("kind").focus({preventScroll: true});
}

function submit(id, action) {
  $(id).addEventListener("submit", async (event) => {
    event.preventDefault();
    const control = event.submitter;
    if (control) control.disabled = true;
    try {
      message();
      await action();
    } catch (error) {
      if (error.status === 401 && id !== "login-form") showLogin();
      message(error.message, true);
    } finally {
      if (control) control.disabled = false;
    }
  });
}

submit("login-form", async () => {
  await api("login", {key: $("key").value});
  $("key").value = "";
  await load();
});
submit("recommend-form", async () => {
  const result = await api("recommend", {
    project: state.project.id, count: $("batch-size").valueAsNumber, request: crypto.randomUUID(),
    strategy: $("bench-strategy").value,
  });
  route(result.project, result.batch);
  await load();
  message("Recommendation batch created.");
});
submit("rename-form", async () => {
  await api("rename", {project: state.project.id, batch: activeBatch().id, name: $("batch-name").value});
  await load({keepSelection: true});
});
submit("settings-form", async () => {
  await api("settings", {
    project: state.project.id, budget: $("budget").valueAsNumber, notes: $("notes").value,
    notes_revision: state.notes.revision, prior_budget: state.project.budget,
  });
  await load({keepSelection: true});
  message("Project settings saved.");
});
submit("result-form", async () => {
  const kind = $("kind").value;
  const status = $("result-status").value;
  const points = kind === "dose_response" && status === "measured"
    ? $("points").value.trim().split(/\n/).map((line) => line.trim().split(/[,\s]+/).map(Number)) : [];
  const revision = editing;
  await api("result", {
    project: state.project.id, candidate: selectedId, request: resultRequest,
    measurement: revision?.id, revision: revision?.revision, kind, unit: $("unit").value,
    status, value: status === "measured" && kind !== "dose_response" ? $("value").valueAsNumber : null,
    points, qualifier: kind === "ic50" && status === "measured" ? $("qualifier").value : "exact",
    charge: $("charge").checked, notes: $("result-notes").value,
    context: {
      species: $("species").value, cell_line: $("cell-line").value, method: $("method").value,
      route: $("route").value, dose_nm: $("dose").valueAsNumber || null,
      hours: $("hours").valueAsNumber || null, detail: $("assay-detail").value,
    },
  });
  await load({keepSelection: true});
  message(revision ? "Correction saved as a new revision." : "Measurement saved.");
});

$("projects").addEventListener("change", () => { message(); route($("projects").value); load(); });
$("add-result").addEventListener("click", () => editResult());
$("kind").addEventListener("change", syncEditor);
$("result-status").addEventListener("change", syncEditor);
$("cancel-result").addEventListener("click", () => { $("result-form").hidden = true; });
$("rename").addEventListener("click", () => {
  $("batch-name").value = activeBatch().name;
  $("rename-form").hidden = false;
  $("batch-name").focus();
});
$("rename-cancel").addEventListener("click", () => { $("rename-form").hidden = true; });
$("logout").addEventListener("click", async () => {
  try {
    await api("logout", {});
    sessionStorage.removeItem(HANDOFF);
    location.replace("/bench");
  } catch (error) {
    message(error.message, true);
  }
});
$("candidate-structure").addEventListener("toggle", () => {
  if ($("candidate-structure").open && selectedId) {
    const row = activeBatch().candidates.find((candidate) => candidate.candidate_id === selectedId);
    $("bench-molecule").show(row, state.project.transcript);
  } else {
    $("bench-molecule").clear();
  }
});
$("bench-strategy").addEventListener("change", () => {
  const policy = $("bench-strategy").value === "ppo";
  $("batch-size").value = policy ? 1 : 5;
  $("batch-size").readOnly = policy;
});
fetch("/api/meta").then((response) => response.json()).then((meta) => {
  $("strategy-field").hidden = !meta.research_models?.length;
}).catch(() => {});
load();
