import {button, displayValue, el, link, notices, plot, shortTime, table} from "./presentation.js";

const pick = selector => document.querySelector(selector);
let lotCatalogue = [];
let extraGeneration = 0;

function itemOptions(select, entries, key, title, value) {
  select.replaceChildren(...entries.map(entry => el("option", {value: entry[key], text: entry[title]})));
  if (value && entries.some(entry => String(entry[key]) === String(value))) select.value = value;
}

function chooseRecipeRows(state) {
  const ready = state.runs.filter(item => item.revision === state.current.revision);
  const kinds = new Set();
  return ready.filter(run => {
    if (kinds.has(run.kind)) return false;
    kinds.add(run.kind);
    return true;
  });
}

function recipeRunInputs(state) {
  const chosen = chooseRecipeRows(state).map((run, index) => el("label", {className: "checkbox"},
    el("input", {type: "checkbox", value: run.id, checked: index === 0}),
    document.createTextNode(run.title + " · " + shortTime(run.created_at)),
  ));
  pick("#recipe-runs").replaceChildren(...(chosen.length ? chosen : [
    el("p", {className: "result-empty", text: "Run a measurement on the current revision first."}),
  ]));
  pick("#recipe-create button").disabled = !chosen.length;
}

export async function enterRecipes(app) {
  const epoch = ++extraGeneration;
  const state = app.state;
  pick("#recipe-subtitle").textContent = state.current.title + " · Revision " + state.current.revision;
  recipeRunInputs(state);
  const field = pick("#recipe-replay").elements.recipe;
  const current = field.value;
  const result = await app.api("/api/recipes");
  if (epoch !== extraGeneration) return;
  itemOptions(field, result.items, "id", "title", current);
  pick("#recipe-replay").querySelectorAll("button").forEach(button => { button.disabled = !result.items.length; });
  if (!result.items.length) {
    pick("#recipe-summary").replaceChildren(el("p", {className: "result-empty", text: "Saved recipes appear here."}));
  }
}

function metricOptions() {
  const form = pick("#lots-form");
  const recipe = lotCatalogue.find(entry => entry.kind === form.elements.kind.value);
  if (!recipe) return;
  itemOptions(form.elements.metric, recipe.readings, "field", "title", form.elements.metric.value);
  metricForm();
}

function metricForm() {
  const form = pick("#lots-form");
  const catalogue = lotCatalogue.find(entry => entry.kind === form.elements.kind.value);
  const metric = catalogue?.readings.find(entry => entry.field === form.elements.metric.value);
  const angular = metric?.mode === "°";
  form.elements.below.disabled = angular;
  form.elements.above.disabled = angular;
  pick("#lot-unit").textContent = angular ? "Use Grain orientation's axis tolerance for directional limits."
    : metric?.mode === "%" ? "Enter screening limits as fractions (0–1)." : "Enter screening limits in the selected scale.";
}

export async function enterLots(app) {
  const epoch = ++extraGeneration;
  const response = await app.api("/api/lots");
  if (epoch !== extraGeneration) return;
  lotCatalogue = response.tools;
  const form = pick("#lots-form");
  const desiredLot = form.elements.lot.value || app.state.current.lot;
  itemOptions(form.elements.lot, response.items, "lot", "lot", desiredLot);
  itemOptions(form.elements.kind, response.tools, "kind", "title", form.elements.kind.value);
  metricOptions();
  form.querySelector("button").disabled = !response.items.length;
  if (!response.items.length) {
    pick("#lot-results").replaceChildren(el("p", {className: "result-empty", text: "Assign at least one face to a lot to build a survey."}));
  }
}

function recipeOutput(outcome, replayed) {
  const rows = outcome.steps.map(step => [
    step.title,
    step.notices.join(" ") || "Saved thresholds are compatible with the target scale.",
  ]);
  const notes = replayed ? outcome.runs.length + " measurements saved." : "Confirm the settings before running.";
  pick("#recipe-summary").replaceChildren(el("p", {text: notes}), table(["Tool", "Position and scale"], rows));
}

function surveyOutput(data, onSelect) {
  const source = data.context;
  const summary = source.summary;
  const labels = [
    ["Lot faces", source.faces],
    ["Numeric readings", summary.usable],
    ["Median", summary.median === null ? "—" : displayValue(summary.median, source.mode, source.units === "metric" ? "mm" : "px")],
  ];
  if (summary.mean_axis !== null) labels.push(["Mean axis", displayValue(summary.mean_axis, "°")]);
  if (summary.concentration !== null) labels.push(["Concentration", displayValue(summary.concentration, "%")]);
  const metrics = el("div", {className: "metrics"}, ...labels.map(([label, number]) => el("div", {className: "metric"},
    el("span", {text: label}), el("strong", {text: number}),
  )));
  const links = el("div", {className: "panel-actions"},
    link("Contact sheet ↓", data.links.sheet), link("CSV ↓", data.links.csv),
  );
  const names = {
    missing: "No eligible measurement", incompatible_scale: "Other scale", undefined: "No numeric reading",
    measured: "Measured", above: "Above limit", below: "Below limit", historic: "Older revision",
  };
  const display = table(
    ["Face", "Revision", source.metric_title, "Measurable area", "Screening"],
    data.items.map(record => [
      record.title, record.revision, record.display, record.measured_area_display || "—",
      names[record.status] || record.status,
    ]),
  );
  display.querySelectorAll("tbody tr").forEach((row, index) => {
    const item = data.items[index];
    row.cells[0].replaceChildren(button(item.title, () => onSelect(item.inspection_id), "text-link"));
  });
  pick("#lot-results").replaceChildren(
    metrics, links, ...(data.notices.length ? [notices(data.notices)] : []), display,
  );
}

export async function showProbe(app, position) {
  const current = app.state.current;
  const path = "/api/inspections/" + current.id;
  const query = new URLSearchParams({x: position[0], y: position[1], revision: current.revision, radius: 12, scale: 8});
  const data = await app.api(path + "/probe?" + query);
  if (app.state.current.id !== current.id) return;
  const coords = data.source_point ? data.source_point.map(value => value.toFixed(2)).join(", ") : "Unavailable";
  const text = [
    ["Measuring pixel", data.center.join(", ")],
    ["Source coordinate", coords],
    ["RGB", data.rgb ? data.rgb.join(", ") : "Outside selected region or source support"],
    ["Usable pixels", displayValue(data.statistics.coverage, "%")],
  ];
  pick("#probe-body").replaceChildren(
    el("div", {className: "probe-content"}, el("img", {src: path + "/probe.png?" + query, alt: "Magnified local pixel neighborhood"}),
      el("dl", {}, ...text.flatMap(([title, value]) => [el("dt", {text: title}), el("dd", {text: value})])),
    ),
    plot(data.horizontal), plot(data.vertical), link("Export projection and pixel values ↓", path + "/probe.csv?" + query),
  );
  app.showDialog("#probe-dialog");
}

export function wireExtras(app) {
  const {api, bindForm, json, notify, showDialog, state} = app;
  const lotForm = pick("#lots-form");
  lotForm.elements.kind.addEventListener("change", metricOptions);
  lotForm.elements.metric.addEventListener("change", metricForm);
  pick("#evidence-open").addEventListener("click", () => {
    pick("#evidence-form").reset();
    pick("#evidence-output").replaceChildren();
    showDialog("#evidence-dialog");
  });
  bindForm("#evidence-form", async form => {
    const checked = await api("/api/evidence/check", {method: "POST", body: new FormData(form)});
    const summary = checked.verified ? "Bundle checks passed." : "At least one file failed its checksum.";
    pick("#evidence-output").replaceChildren(
      el("strong", {text: summary + " " + checked.report_title}),
      el("p", {className: "subtle", text: checked.notice}),
      table(["Evidence file", "Bytes", "Checksum"], checked.files.map(item => [item.name, item.size, item.verified ? "Matched" : "Mismatch"])),
    );
  });
  bindForm("#recipe-create", async form => {
    const current = state.current;
    const data = Object.fromEntries(new FormData(form));
    const ids = [...form.querySelectorAll('input[type="checkbox"]:checked')].map(item => item.value);
    await api("/api/recipes", json("POST", {...data, inspection_id: current.id, revision: current.revision, run_ids: ids}));
    form.reset();
    await enterRecipes(app);
    notify("Recipe saved");
  });
  bindForm("#recipe-replay", async (form, trigger) => {
    const current = state.current;
    const recipe = form.elements.recipe.value;
    const mode = trigger?.value ?? "prepare";
    const address = "/api/recipes/" + encodeURIComponent(recipe) + "/" + mode;
    const payload = {inspection_id: current.id, revision: current.revision, positions: form.elements.positions.value};
    const result = await api(address, json("POST", payload));
    if (current.id !== state.current.id) return;
    if (mode === "run") {
      state.runs.unshift(...result.runs);
      recipeRunInputs(state);
      notify("Recipe measurements saved");
    }
    recipeOutput(result, mode === "run");
  });
  bindForm("#lots-form", async form => {
    const data = new FormData(form);
    for (const key of ["below", "above"]) if (!data.get(key)) data.delete(key);
    const result = await api("/api/lots/survey?" + new URLSearchParams(data));
    surveyOutput(result, id => app.openInspection(id, true).catch(error => notify(error.message, true)));
  });
}
