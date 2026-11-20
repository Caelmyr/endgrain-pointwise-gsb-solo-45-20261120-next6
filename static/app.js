import {WorkbenchViewer} from "./viewer.js";
import {enterLots, enterRecipes, showProbe, wireExtras} from "./extras.js";
import {
  button, createToolInput, displayValue, el, link, metricCards, notices,
  objectTable, plot, shortTime, table,
} from "./presentation.js";

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const state = {
  library: [], tools: new Map(), current: null, stats: null,
  runs: [], notes: [], selectedRun: null, tool: "rings", settings: {},
  view: "inspect", image: "working", drawing: null, marker: null,
  transect: null, swatches: null, footprintSeed: null, edgeCorners: null,
  selectionEpoch: 0, queryEpoch: 0,
};

const viewer = new WorkbenchViewer(coordinate);
let noticeTimer;

function currentApi(tail = "") {
  return "/api/inspections/" + state.current.id + tail;
}

async function api(endpoint, options = {}) {
  const response = await fetch(endpoint, options);
  if (response.status === 204) return null;
  if (!response.ok) {
    const message = await response.json().catch(() => ({}));
    const error = new Error(message.error || "The operation could not be completed.");
    error.status = response.status;
    throw error;
  }
  return await response.json();
}

function json(method, body) {
  return {method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
}

function notify(text, bad = false) {
  const toast = $("#toast");
  toast.textContent = text;
  toast.hidden = false;
  toast.classList.toggle("error", bad);
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { toast.hidden = true; }, bad ? 5200 : 2900);
}

function status(text = "", bad = false) {
  const element = $("#status-bar");
  element.textContent = text;
  element.hidden = !text;
  element.classList.toggle("error", bad);
}

function enablePending(form, active) {
  form.querySelectorAll('button[type="submit"]').forEach(control => { control.disabled = active; });
}

function bindForm(selector, handler) {
  const form = $(selector);
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const container = form.querySelector(".dialog-error");
    if (container) container.hidden = true;
    enablePending(form, true);
    try {
      await handler(form, event.submitter);
    } catch (error) {
      if (container) {
        container.hidden = false;
        container.textContent = error.message;
      } else {
        if (state.view === "inspect") status(error.message, true);
        notify(error.message, true);
      }
    } finally {
      enablePending(form, false);
    }
  });
}

function showDialog(selector) {
  const dialog = $(selector);
  const error = dialog.querySelector(".dialog-error");
  if (error) error.hidden = true;
  if (!dialog.open) dialog.showModal();
}

function renderLibrary() {
  const target = $("#sample-list");
  if (!state.library.length) {
    target.replaceChildren(el("p", {className: "empty-library", text: "No inspections found."}));
    return;
  }
  target.replaceChildren(...state.library.map(record => {
    const active = record.id === state.current?.id;
    return el("button", {
      type: "button", className: "sample-card" + (active ? " active" : ""),
      onclick: () => { openInspection(record.id).catch(error => notify(error.message, true)); },
      "aria-pressed": String(active),
    },
    el("img", {src: record.links.thumb, alt: ""}),
    el("span", {},
      el("span", {className: "sample-title", text: record.title}),
      el("span", {className: "sample-meta", text: [record.capture_metadata.synthetic ? "Synthetic" : record.surface, record.lot && "Lot " + record.lot].filter(Boolean).join(" · ")}),
    ));
  }));
}

async function reloadLibrary() {
  const generation = ++state.queryEpoch;
  const filter = $("#search").value;
  const data = await api("/api/inspections?" + new URLSearchParams({q: filter}));
  if (generation !== state.queryEpoch) return;
  state.library = data.items;
  renderLibrary();
}

function clearGeometry() {
  state.drawing = null;
  state.transect = null;
  state.swatches = null;
  state.footprintSeed = null;
  state.edgeCorners = null;
  state.marker = null;
  $("#manual-crossings").value = "";
}

async function openInspection(inspectionId, activate = false) {
  const epoch = ++state.selectionEpoch;
  status();
  const path = "/api/inspections/" + inspectionId;
  const [record, runData, notes] = await Promise.all([api(path), api(path + "/runs"), api(path + "/notes")]);
  if (epoch !== state.selectionEpoch) return;
  if (state.current?.id !== inspectionId || state.current.revision !== record.revision) {
    clearGeometry();
    viewer.resetZoom();
  }
  state.current = record;
  state.stats = null;
  state.runs = runData.items;
  state.notes = notes.items;
  state.selectedRun = state.runs.find(item => item.revision === record.revision) || null;
  state.image = "working";
  if (activate) state.view = "inspect";
  $("#report-title").value = record.title + " inspection";
  $("#compare-output").replaceChildren();
  renderAll();
  api(path + "/stats").then(loaded => {
    if (epoch !== state.selectionEpoch) return;
    state.stats = loaded;
    renderSetup();
  }).catch(error => notify(error.message, true));
  enterPanel();
}

function setView(view) {
  state.view = view;
  if (view !== "inspect") state.drawing = null;
  renderAll();
  enterPanel();
}

function extrasContext() {
  return {state, api, bindForm, json, notify, showDialog, openInspection};
}

function enterPanel() {
  if (!state.current) return;
  if (state.view === "reports") renderReports().catch(error => notify(error.message, true));
  if (state.view === "recipes") enterRecipes(extrasContext()).catch(error => notify(error.message, true));
  if (state.view === "lots") enterLots(extrasContext()).catch(error => notify(error.message, true));
}

function renderAll() {
  const ready = !!state.current;
  const active = ready ? state.view : "inspect";
  $$(".main-nav [data-view]").forEach(item => {
    item.classList.toggle("active", item.dataset.view === active);
    item.disabled = !ready && item.dataset.view !== "inspect";
  });
  $("#empty").hidden = ready;
  ["inspect", "compare", "reports", "recipes", "lots"].forEach(key => { $("#" + key + "-view").hidden = !ready || active !== key; });
  $("#controls").hidden = !ready || active !== "inspect";
  renderLibrary();
  if (!ready) return;
  if (active === "inspect") {
    renderHeader();
    renderSetup();
    renderTools();
    renderHistory();
    renderResults();
    renderNotes();
    renderImage();
  } else if (active === "compare") {
    renderCompareSelectors();
  } else if (active === "reports") {
    renderReportInputs();
  }
}

function renderHeader() {
  const record = state.current;
  const meta = record.capture_metadata;
  $("#face-kicker").textContent = meta.synthetic ? "SYNTHETIC SAMPLE" : "INSPECTION";
  $("#face-title").textContent = record.title;
  const labels = [
    meta.demo_hint || [record.material, record.lot && "Lot " + record.lot, record.surface].filter(Boolean).join(" · "),
    meta.format + " " + meta.width + " × " + meta.height,
  ];
  $("#face-description").textContent = labels.join("  /  ");
}

function renderSetup() {
  if (!state.current) return;
  const cur = state.current;
  const plane = cur.plane;
  $("#revision-chip").textContent = "Revision " + cur.revision;
  const calibration = {
    pixels: "Pixel dimensions",
    ruler: "Ruler calibrated",
    quadrilateral: "Rectified plane",
  };
  const clear = (name, callback) => button(name, callback, "text-link");
  const stats = state.stats;
  const selected = cur.state.region ? "Selected inspection region" : "Whole image";
  $("#setup-state").replaceChildren(
    el("div", {className: "setup-row"}, el("span", {text: calibration[plane.calibration.mode]}),
      el("strong", {text: plane.unit === "mm" ? (plane.mm_per_x.toFixed(3) + " mm/px") : "px"})),
    el("div", {className: "setup-row"}, el("span", {text: selected}),
      ...(cur.state.region ? [clear("Clear", () => change({region: null}, "Region cleared").catch(error => status(error.message, true)))] : [])),
    el("div", {className: "setup-row"}, el("span", {text: cur.state.lighting ? "Reference applied" : "Original illumination"}),
      ...(stats?.lighting ? [el("strong", {text: displayValue(stats.lighting.reference_coverage, "%")})] : [])),
  );
  if (stats && stats.coverage !== null) {
    $("#setup-state").append(el("div", {className: "setup-row"}, el("span", {text: "Supported area"}),
      el("strong", {text: displayValue(stats.coverage, "%")})));
  }
  const drawing = $("#drawing-actions");
  drawing.hidden = !state.drawing;
  if (state.drawing) {
    const count = state.drawing.points.length;
    const apply = button("Use points", () => finishDrawing().catch(error => status(error.message, true)), "button strong");
    apply.disabled = count < state.drawing.minimum || count > state.drawing.maximum;
    const undo = button("Undo point", () => {
      state.drawing.points.pop();
      renderImage();
      renderSetup();
    });
    undo.disabled = count === 0;
    drawing.replaceChildren(apply, undo, button("Cancel", cancelDrawing, "button ghost"));
  }
}

function currentTool() {
  return state.tools.get(state.tool);
}

function renderTools() {
  const tools = [...state.tools.values()];
  const select = $("#tool-select");
  if (!select.options.length) select.replaceChildren(...tools.map(({id, title}) => el("option", {value: id, text: title})));
  if (!currentTool()) return;
  select.value = state.tool;
  $("#tool-description").textContent = currentTool().description;
  const values = state.settings[state.tool] || {};
  $("#tool-inputs").replaceChildren(...currentTool().inputs.map(spec => {
    const wrapper = createToolInput(spec, values[spec.key]);
    const input = wrapper.querySelector("input,select");
    input.addEventListener("input", () => {
      state.settings[state.tool] ??= {};
      state.settings[state.tool][spec.key] = input.type === "checkbox" ? input.checked
        : input.type === "number" ? input.valueAsNumber : input.value;
    });
    return wrapper;
  }));
  renderToolGeometry();
}

function renderToolGeometry() {
  const geometries = {
    rings: ["Place transect", "Mark the inner and outer ends in order.", 2, 2, state.transect],
    edge_loss: ["Mark expected outline", "Place four corners clockwise from the upper left.", 4, 4, state.edgeCorners],
    colors: ["Place color swatches", "Place 2–4 centers for the color reference patches.", 2, 4, state.swatches],
    footprint: ["Choose wood component", "Choose one point inside the wood face to isolate.", 1, 1, state.footprintSeed ? [state.footprintSeed] : null],
  };
  const group = $("#geometry-controls");
  group.replaceChildren();
  $("#ring-manual").hidden = state.tool !== "rings";
  $("#tool-hint").textContent = "Each run saves its overlay, settings and measurement table.";
  if (state.tool in geometries) {
    const [name, hint, minimum, maximum, placed] = geometries[state.tool];
    const options = [button(name, () => startDrawing(state.tool, hint, minimum, maximum, "working"))];
    if (placed) options.push(button("Clear", () => {
      if (state.tool === "rings") state.transect = null;
      if (state.tool === "colors") state.swatches = null;
      if (state.tool === "footprint") state.footprintSeed = null;
      if (state.tool === "edge_loss") state.edgeCorners = null;
      renderToolGeometry();
      renderImage();
    }, "text-link"));
    group.append(el("div", {className: "geometry-inline"}, ...options));
    const defaults = {
      rings: "You can use the default transect, or mark the line to measure.",
      edge_loss: "Mark the nominal shape or start with the default outline.",
      colors: "Choose color swatches on supported parts of your sample.",
      footprint: "Select a wood component, or use the largest detected silhouette.",
    };
    $("#tool-hint").textContent = placed ? placed.length + " marked position(s)." : defaults[state.tool];
  }
}

function renderImage() {
  const record = state.current;
  if (!record) return;
  const showResult = state.image === "result" && state.selectedRun;
  const showGain = state.image === "gain" && record.state.lighting;
  const resultRun = state.selectedRun;
  let mode = state.image;
  let url = record.links.working + "?r=" + record.revision;
  let title = "Working image";
  if (mode === "source") {
    url = record.links.source;
    title = "Original capture";
  }
  if (mode === "grid") {
    url = currentApi("/calibration.png?r=" + record.revision);
    title = "Calibrated measuring grid";
  }
  if (showResult) {
    url = resultRun.links.preview;
    title = resultRun.title + " overlay";
  }
  if (showGain) {
    url = currentApi("/lighting.png?view=gain&r=" + record.revision);
    title = "Illumination gain map";
  }
  if ((mode === "result" && !showResult) || (mode === "gain" && !showGain)) mode = "working";
  const plane = showResult ? resultRun.snapshot.plane : record.plane;
  $$("[data-image]").forEach(item => {
    item.classList.toggle("active", item.dataset.image === mode);
    item.disabled = (item.dataset.image === "result" && !state.selectedRun)
      || (item.dataset.image === "gain" && !record.state.lighting)
      || (!!state.drawing && item.dataset.image !== mode);
  });
  const sizes = ["source", "gain"].includes(mode) ? [plane.source_width, plane.source_height] : [plane.width, plane.height];
  $("#plane-label").textContent = sizes.join(" × ") + " px" + (mode === "working" || mode === "result" ? " · " + plane.unit : "");
  viewer.show({
    url, title, mode, plane, drawing: state.drawing, region: record.state.region,
    notes: showResult && resultRun.revision !== record.revision ? [] : state.notes,
    guide: state.tool === "rings" ? state.transect : null,
  });
}

function renderHistory() {
  const list = $("#run-list");
  $("#run-count").textContent = state.runs.length ? String(state.runs.length) : "";
  if (!state.runs.length) {
    list.replaceChildren(el("p", {className: "result-empty", text: "No saved measurements yet."}));
    return;
  }
  list.replaceChildren(...state.runs.map(run => {
    const classes = "run-item" + (state.selectedRun?.id === run.id ? " active" : "");
    return el("button", {type: "button", className: classes, onclick: () => selectRun(run)},
      el("strong", {text: run.title}),
      el("small", {text: "Rev " + run.revision + " · " + shortTime(run.created_at)}),
    );
  }));
}

function selectRun(run) {
  state.drawing = null;
  state.selectedRun = run;
  state.image = "result";
  renderHistory();
  renderResults();
  renderSetup();
  renderImage();
}

function renderResults() {
  const output = $("#result-body");
  const stored = state.selectedRun;
  const downloads = $("#result-downloads");
  downloads.replaceChildren();
  if (!stored) {
    $("#result-title").textContent = "Latest result";
    output.replaceChildren(el("p", {className: "result-empty", text: "Run a measurement to see readings and download its evidence."}));
    return;
  }
  $("#result-title").textContent = stored.title;
  downloads.append(link("Overlay ↓", stored.links.preview + "?download=1"), link("CSV ↓", stored.links.csv));
  const items = [metricCards(state.tools.get(stored.kind), stored.result)];
  if (stored.result.chart) items.push(plot(stored.result.chart));
  const messages = [...stored.result.notices];
  if (stored.revision !== state.current.revision) messages.unshift("This measurement belongs to revision " + stored.revision + ".");
  if (messages.length) items.push(notices(messages));
  const rows = objectTable(stored);
  if (rows) items.push(rows);
  output.replaceChildren(...items);
}

function renderNotes() {
  const nodes = state.notes.map(note => {
    const remove = button("×", async () => {
      try {
        await api(currentApi("/notes/" + note.id), json("DELETE", {revision: state.current.revision}));
        state.notes = state.notes.filter(item => item.id !== note.id);
        renderNotes();
        renderImage();
        notify("Marker removed");
      } catch (error) {
        notify(error.message, true);
      }
    }, "icon-button");
    remove.setAttribute("aria-label", "Remove " + note.label);
    return el("div", {className: "note-item"},
      el("div", {}, el("strong", {text: note.label}), el("p", {text: note.comment})),
      remove,
    );
  });
  $("#notes-list").replaceChildren(...(nodes.length ? nodes : [
    el("p", {className: "notes-empty", text: "Mark a location to keep an observation with this revision."}),
  ]));
}

function startDrawing(kind, hint, minimum, maximum, mode, extra = {}) {
  state.image = mode;
  state.drawing = {kind, hint, minimum, maximum, points: [], ...extra};
  status();
  renderSetup();
  renderImage();
}

function coordinate(point) {
  const drawing = state.drawing;
  if (!drawing || drawing.points.length >= drawing.maximum) return;
  drawing.points.push(point);
  renderSetup();
  renderImage();
  if (["marker", "probe"].includes(drawing.kind)) finishDrawing().catch(error => notify(error.message, true));
}

function cancelDrawing() {
  state.drawing = null;
  renderSetup();
  renderImage();
}

async function finishDrawing() {
  const draw = state.drawing;
  if (!draw || draw.points.length < draw.minimum) return;
  const items = draw.points;
  if (draw.kind === "region") {
    await change({region: items}, "Inspection region saved");
    return;
  }
  if (draw.kind === "calibration") {
    const calibration = {...draw.calibration};
    if (calibration.mode === "ruler") [calibration.a, calibration.b] = items;
    else calibration.corners = items;
    await change({calibration}, "Calibration saved");
    return;
  }
  if (draw.kind === "rings") state.transect = items;
  if (draw.kind === "colors") state.swatches = items;
  if (draw.kind === "footprint") state.footprintSeed = items[0];
  if (draw.kind === "edge_loss") state.edgeCorners = items;
  if (draw.kind === "probe") await showProbe(extrasContext(), items[0]);
  if (draw.kind === "marker") {
    state.marker = items[0];
    $("#marker-form").reset();
    showDialog("#marker-dialog");
  }
  state.drawing = null;
  renderToolGeometry();
  renderSetup();
  renderImage();
}

async function change(patch, message) {
  const id = state.current.id;
  const expected = state.current.revision;
  const delta = await api(currentApi(), json("PATCH", {revision: expected, ...patch}));
  if (state.current?.id !== id) return;
  await openInspection(delta.id);
  reloadLibrary().catch(error => notify(error.message, true));
  notify(message);
}

function settingsFromForm(form) {
  const params = {};
  form.querySelectorAll("[data-setting]").forEach(input => {
    const value = input.type === "checkbox" ? input.checked : input.type === "number" ? input.valueAsNumber : input.value;
    params[input.name] = value;
  });
  if (state.tool === "rings") {
    if (state.transect) params.endpoints = state.transect;
    const manual = $("#manual-crossings").value.trim();
    if (manual) {
      params.manual_crossings = manual.split(",").map(value => Number(value.trim()));
      if (params.manual_crossings.some(value => !Number.isFinite(value))) throw new Error("Separate numeric fractions with commas.");
    }
  }
  if (state.tool === "colors" && state.swatches) {
    params.swatches = state.swatches.map((point, index) => ({label: "Tone " + String.fromCharCode(65 + index), point}));
  }
  if (state.tool === "footprint" && state.footprintSeed) params.seed = state.footprintSeed;
  if (state.tool === "edge_loss" && state.edgeCorners) params.corners = state.edgeCorners;
  return params;
}

function renderCompareSelectors() {
  $("#compare-subtitle").textContent = state.current.title + " · choose two saved measurements of the same kind.";
  const a = $("#compare-a");
  const picked = a.value;
  a.replaceChildren(...state.runs.map(run => el("option", {value: run.id, text: runCaption(run)})));
  if (state.runs.some(run => run.id === picked)) a.value = picked;
  renderCompareSecond();
}

function runCaption(run) {
  return run.title + " · Rev " + run.revision + " · " + shortTime(run.created_at);
}

function renderCompareSecond() {
  const a = $("#compare-a").value;
  const b = $("#compare-b");
  const kind = state.runs.find(run => run.id === a)?.kind;
  b.replaceChildren(...state.runs.filter(run => run.kind === kind && run.id !== a)
    .map(run => el("option", {value: run.id, text: runCaption(run)})));
  $("#compare-form button").disabled = !b.value;
  if (!b.value) $("#compare-output").replaceChildren(el("p", {className: "compare-empty", text: "Save a second measurement of this kind to compare."}));
}

function renderCompareResult(result) {
  const compareUnit = result.left.result.unit;
  const formatted = result.rows.map(row => [
    row.title,
    displayValue(row.left, row.mode, compareUnit, result.left.result.metrics),
    displayValue(row.right, row.mode, result.right.result.unit, result.right.result.metrics),
    displayValue(row.difference, row.mode, compareUnit, result.left.result.metrics),
  ]);
  const image = el("img", {src: result.links.image, className: "compare-image", alt: "Side-by-side overlays for measurements A and B"});
  const buttons = el("div", {className: "panel-actions"}, link("Download comparison CSV ↓", result.links.table));
  if (result.links.delta) {
    let showing = false;
    const toggle = button("Show overlay differences", () => {
      showing = !showing;
      image.src = showing ? result.links.delta : result.links.image;
      image.alt = showing ? "Pixel differences between saved overlays" : "Side-by-side overlays for measurements A and B";
      toggle.textContent = showing ? "Show side by side" : "Show overlay differences";
    });
    buttons.append(toggle);
  }
  $("#compare-output").replaceChildren(
    image, buttons,
    ...(result.notices.length ? [notices(result.notices)] : []),
    table(["Reading", "A", "B", "B − A"], formatted),
  );
}

function renderReportInputs() {
  $("#report-subtitle").textContent = state.current.title + " · Revision " + state.current.revision;
  $("#archive-download").href = currentApi("/archive.zip");
  const ready = state.runs.filter(item => item.revision === state.current.revision);
  const targets = ready.map((run, index) => el("label", {className: "checkbox"},
    el("input", {type: "checkbox", value: run.id, checked: index === 0}),
    document.createTextNode(run.title + " · " + shortTime(run.created_at)),
  ));
  $("#report-run-list").replaceChildren(...(targets.length ? targets : [
    el("p", {className: "notes-empty", text: "Run and save a measurement on the current revision first."}),
  ]));
  $("#report-form button").disabled = ready.length === 0;
}

async function renderReports() {
  const id = state.current.id;
  const result = await api(currentApi("/reports"));
  if (id !== state.current?.id) return;
  const values = result.items.map(report => {
    const printable = link("Print view", report.links.html);
    printable.target = "_blank";
    printable.rel = "noopener";
    const external = link("External review ZIP ↓", report.links.review);
    external.title = "Region-only copy for a commissioning party: no source capture, no illumination reference";
    return el("div", {className: "issued-row"},
      el("div", {}, el("strong", {text: report.title}),
        el("small", {text: "Revision " + report.revision + " · " + report.reviewer + " · " + shortTime(report.created_at)})),
      printable, link("Evidence ZIP ↓", report.links.zip), external,
    );
  });
  $("#issued-reports").replaceChildren(...(values.length ? values : [el("p", {className: "notes-empty", text: "No reports have been issued."})]));
}

async function startDemo() {
  $("#demo-load").disabled = true;
  $("#empty-open").disabled = true;
  notify("Preparing synthetic samples…");
  try {
    const loaded = await api("/api/demo", {method: "POST"});
    $("#search").value = "";
    await reloadLibrary();
    if (loaded.items.length) await openInspection(loaded.items[0].id, true);
    notify("Synthetic demo samples ready");
  } catch (error) {
    notify(error.message, true);
  } finally {
    $("#demo-load").disabled = false;
    $("#empty-open").disabled = false;
  }
}

function calFields() {
  const form = $("#cal-form");
  const method = form.elements.mode.value;
  $("#cal-ruler").hidden = method !== "ruler";
  $("#cal-quad").hidden = method !== "quadrilateral";
  $("#cal-next").textContent = method === "pixels" ? "Use pixels" : "Continue on image";
  $("#cal-hint").textContent = {
    pixels: "Measure in working pixels without a physical scale.",
    ruler: "Choose two visible reference points a known distance apart.",
    quadrilateral: "Place the four corners of a known rectangle clockwise from upper left. Changing the measuring plane clears the old region.",
  }[method];
}

async function openLight() {
  showDialog("#light-dialog");
  const received = await api("/api/references");
  const all = received.items.filter(entry => entry.metadata.width === state.current.capture_metadata.width
    && entry.metadata.height === state.current.capture_metadata.height);
  const form = $("#light-form");
  form.reset();
  const suggestions = $("#reference-picker");
  suggestions.replaceChildren(el("option", {value: "", text: "Choose a matching image"}),
    ...all.map(entry => el("option", {value: entry.id, text: entry.title})));
  const prior = state.current.state.lighting;
  if (prior) Object.entries(prior).forEach(([key, value]) => {
    form.elements[key].value = value;
  });
  else if (all.length === 1) suggestions.value = all[0].id;
}

function wire() {
  $$("[data-close]").forEach(item => item.addEventListener("click", () => item.closest("dialog").close()));
  $$("[data-view]").forEach(item => item.addEventListener("click", () => setView(item.dataset.view)));
  $$("[data-image]").forEach(item => item.addEventListener("click", () => {
    state.image = item.dataset.image;
    renderImage();
  }));
  $("#demo-load").addEventListener("click", startDemo);
  $("#empty-open").addEventListener("click", startDemo);
  const importPhoto = () => showDialog("#import-dialog");
  $("#import-open").addEventListener("click", importPhoto);
  $("#empty-import").addEventListener("click", importPhoto);
  $("#restore-open").addEventListener("click", () => showDialog("#restore-dialog"));
  $("#search").addEventListener("input", () => reloadLibrary().catch(error => notify(error.message, true)));
  $("#marker-start").addEventListener("click", () => startDrawing("marker", "Click a supported point to place a review marker.", 1, 1, "working"));
  $("#probe-start").addEventListener("click", () => startDrawing("probe", "Click a point to inspect its local image detail.", 1, 1, "working"));
  $("#roi-start").addEventListener("click", () => startDrawing("region", "Trace the region with 3–48 vertices, then choose Use points.", 3, 48, "working"));
  $("#details-open").addEventListener("click", () => {
    const inputs = $("#details-form").elements;
    ["title", "material", "lot", "surface"].forEach(key => { inputs[key].value = state.current[key]; });
    showDialog("#details-dialog");
  });
  $("#cal-open").addEventListener("click", () => {
    const form = $("#cal-form");
    form.reset();
    const saved = state.current.state.calibration;
    form.elements.mode.value = saved.mode === "pixels" ? "ruler" : saved.mode;
    if (saved.length_mm) form.elements.length_mm.value = saved.length_mm;
    if (saved.width_mm) form.elements.width_mm.value = saved.width_mm;
    if (saved.height_mm) form.elements.height_mm.value = saved.height_mm;
    calFields();
    showDialog("#cal-dialog");
  });
  $("#cal-form").elements.mode.addEventListener("change", calFields);
  $("#light-open").addEventListener("click", () => openLight().catch(error => notify(error.message, true)));
  $("#light-reset").addEventListener("click", async () => {
    try {
      await change({lighting: null}, "Original illumination restored");
      $("#light-dialog").close();
    } catch (error) {
      notify(error.message, true);
    }
  });
  $("#tool-select").addEventListener("change", event => {
    state.tool = event.target.value;
    if (state.drawing) cancelDrawing();
    renderTools();
    renderImage();
  });
  $("#compare-a").addEventListener("change", renderCompareSecond);
  bindForm("#import-form", async form => {
    const created = await api("/api/inspections", {method: "POST", body: new FormData(form)});
    form.reset();
    $("#import-dialog").close();
    $("#search").value = "";
    await reloadLibrary();
    await openInspection(created.id, true);
    notify("Photograph imported");
  });
  bindForm("#restore-form", async form => {
    const restored = await api("/api/inspections/restore", {method: "POST", body: new FormData(form)});
    form.reset();
    $("#restore-dialog").close();
    $("#search").value = "";
    await reloadLibrary();
    await openInspection(restored.id, true);
    notify("Inspection restored");
  });
  bindForm("#details-form", async form => {
    await change(Object.fromEntries(new FormData(form)), "Inspection details saved");
    $("#details-dialog").close();
  });
  bindForm("#cal-form", async form => {
    const fields = form.elements;
    const mode = fields.mode.value;
    const calibration = {mode};
    if (mode === "pixels") {
      await change({calibration}, "Working in pixels");
      $("#cal-dialog").close();
      return;
    }
    if (mode === "ruler") {
      calibration.length_mm = fields.length_mm.valueAsNumber;
      if (!Number.isFinite(calibration.length_mm)) throw new Error("Enter the measured length.");
      startDrawing("calibration", "Mark the two points of the ruler and choose Use points.", 2, 2, "source", {calibration});
    } else {
      calibration.width_mm = fields.width_mm.valueAsNumber;
      calibration.height_mm = fields.height_mm.valueAsNumber;
      if (!Number.isFinite(calibration.width_mm) || !Number.isFinite(calibration.height_mm)) throw new Error("Enter the rectangle width and height.");
      startDrawing("calibration", "Mark four corners clockwise, starting upper left, then choose Use points.", 4, 4, "source", {calibration});
    }
    $("#cal-dialog").close();
  });
  bindForm("#light-form", async form => {
    const inputs = form.elements;
    let selected = inputs.reference_id.value;
    if (inputs.file.files.length) {
      const data = new FormData();
      data.append("file", inputs.file.files[0]);
      const uploaded = await api("/api/references", {method: "POST", body: data});
      selected = uploaded.id;
    }
    if (!selected) throw new Error("Choose or import an illumination reference.");
    const values = {reference_id: selected};
    ["strength", "max_gain", "blur_px"].forEach(key => { values[key] = inputs[key].valueAsNumber; });
    await change({lighting: values}, "Illumination correction saved");
    $("#light-dialog").close();
    state.image = "gain";
    renderImage();
  });
  bindForm("#marker-form", async form => {
    const payload = {revision: state.current.revision, point: state.marker, ...Object.fromEntries(new FormData(form))};
    const saved = await api(currentApi("/notes"), json("POST", payload));
    state.notes.push(saved);
    state.marker = null;
    $("#marker-dialog").close();
    renderNotes();
    renderImage();
    notify("Marker saved");
  });
  bindForm("#tool-form", async form => {
    const inspection = state.current.id;
    const payload = {
      revision: state.current.revision, kind: state.tool, params: settingsFromForm(form),
    };
    status("Measuring the current inspection region…");
    const run = await api(currentApi("/runs"), json("POST", payload));
    if (state.current?.id !== inspection) return;
    state.runs.unshift(run);
    selectRun(run);
    status("Saved on revision " + run.revision + ".");
    notify(run.title + " saved");
  });
  bindForm("#compare-form", async () => {
    const params = new URLSearchParams({left: $("#compare-a").value, right: $("#compare-b").value});
    const result = await api(currentApi("/compare?") + params);
    renderCompareResult(result);
  });
  bindForm("#report-form", async () => {
    const selected = $$("#report-run-list input:checked").map(input => input.value);
    const body = {
      title: $("#report-title").value, reviewer: $("#report-reviewer").value,
      disposition: $("#report-disposition").value, summary: $("#report-summary").value,
      run_ids: selected, revision: state.current.revision,
    };
    await api(currentApi("/reports"), json("POST", body));
    await renderReports();
    notify("Report issued");
  });
}

async function boot() {
  wire();
  wireExtras(extrasContext());
  const catalogue = await api("/api/tools");
  state.tools = new Map(catalogue.tools.map(tool => [tool.id, tool]));
  await reloadLibrary();
  if (state.library.length) await openInspection(state.library[0].id);
  else renderAll();
}

boot().catch(error => notify(error.message, true));
