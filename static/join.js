import {button, el, notices, table} from "./presentation.js";

const pick = selector => document.querySelector(selector);
let joinGeneration = 0;
let joinState = {items: [], result: null, candidate: null, view: "image", app: null};

function frameLabel(entry) {
  return entry.title + " · " + entry.width + "×" + entry.height + (entry.joined ? " · joined" : "");
}

function fillFrames(items) {
  const first = pick("#join-first");
  const second = pick("#join-second");
  const priorFirst = first.value;
  const priorSecond = second.value;
  const options = items.map(entry => el("option", {value: entry.id, text: frameLabel(entry)}));
  first.replaceChildren(...options.map(node => node.cloneNode(true)));
  second.replaceChildren(el("option", {value: "", text: "Choose the second frame"}), ...options);
  if (items.some(entry => entry.id === priorFirst)) first.value = priorFirst;
  if (items.some(entry => entry.id === priorSecond)) second.value = priorSecond;
  else if (items.length > 1 && second.value === "") {
    second.value = items.find(entry => entry.id !== first.value)?.id ?? "";
  }
}

async function loadFrames(app) {
  const response = await app.api("/api/captures");
  joinState.items = response.items;
  fillFrames(response.items);
}

function joinPayload() {
  const form = pick("#join-form");
  const data = new FormData(form);
  const payload = {
    first: data.get("first"),
    second: data.get("second"),
    side: data.get("side"),
    title: data.get("title") || "",
    material: data.get("material") || "",
    lot: data.get("lot") || "",
    surface: data.get("surface") || "unknown",
    seam: {
      focus_margin: Number(data.get("focus_margin")),
      match_brightness: data.get("match_brightness") === "on",
    },
  };
  return payload;
}

function showJoinStatus(text, bad = false) {
  const node = pick("#join-status");
  node.textContent = text;
  node.hidden = !text;
  node.classList.toggle("error", bad);
}

function chosenCandidate() {
  return joinState.result?.candidates?.find(
    item => item.dx === joinState.candidate?.dx && item.dy === joinState.candidate?.dy) ?? null;
}

function previewUrl(candidate, view) {
  if (!candidate) return "";
  const query = new URLSearchParams({
    first: joinState.result.first.capture_id,
    second: joinState.result.second.capture_id,
    side: joinState.result.side,
    dx: String(candidate.dx),
    dy: String(candidate.dy),
    view,
  });
  const seam = previewSeam();
  query.set("focus_margin", String(seam.focus_margin));
  query.set("match_brightness", seam.match_brightness ? "1" : "0");
  return "/api/stitches/preview.png?" + query.toString();
}

function previewSeam() {
  return joinPayload().seam;
}

function renderJoinCandidates() {
  const target = pick("#join-candidates");
  const result = joinState.result;
  if (!result?.candidates?.length) {
    target.replaceChildren(el("p", {className: "result-empty", text: "No overlap was found for this side."}));
    return;
  }
  target.replaceChildren(...result.candidates.map((item, index) => {
    const active = joinState.candidate?.dx === item.dx && joinState.candidate?.dy === item.dy;
    const radio = el("input", {
      type: "radio", name: "join-candidate", value: String(index), checked: active,
      onchange: () => { joinState.candidate = item; renderJoinCandidates(); renderJoinPreview(); },
    });
    const heading = el("strong", {text: "Offset " + item.dx + ", " + item.dy + " · score " + item.score.toFixed(3)});
    if (item.ambiguous) heading.append(el("i", {className: "flag", text: "check this one"}));
    const detail = [
      "Overlap " + item.overlap_width + "×" + item.overlap_height + " px",
      item.reason || (index === 0 ? "strongest match" : "weaker alternative"),
    ].join(" · ");
    return el("label", {className: "join-candidate" + (active ? " active" : "")},
      radio, el("span", {}, heading, el("small", {text: detail})),
    );
  }));
}

function renderJoinPreview() {
  const image = pick("#join-image");
  const candidate = chosenCandidate();
  if (!candidate) {
    image.removeAttribute("src");
    pick("#join-seam-notes").replaceChildren();
    return;
  }
  image.src = previewUrl(candidate, joinState.view);
  image.alt = joinState.view === "seam"
    ? "Seam source map: blue from the first frame, amber from the second, pink where the choice is uncertain"
    : "Preview of the joined frames";
  pick("[data-join-view='image']").classList.toggle("active", joinState.view === "image");
  pick("[data-join-view='seam']").classList.toggle("active", joinState.view === "seam");
  renderSeamNotes(candidate);
}

async function renderSeamNotes(candidate) {
  const generation = ++joinGeneration;
  const query = previewUrl(candidate, "image").split("?")[1];
  try {
    const detail = await joinState.app.api("/api/stitches/preview.json?" + query);
    if (generation !== joinGeneration) return;
    renderSeamDetail(detail);
  } catch (error) {
    if (generation !== joinGeneration) return;
    pick("#join-seam-notes").replaceChildren();
  }
}

function renderSeamDetail(detail) {
  const seam = detail.seam;
  const first = seam.sources.first.title;
  const second = seam.sources.second.title;
  const cards = [
    ["First frame", first, "Kept on its own side of the seam."],
    ["Second frame", second, "Moved by " + seam.placement.second.join(", ") + " pixels."],
    ["Hard to call", detail.uncertain_percent + "%", "No clear sharper frame; the nearer frame keeps these pixels."],
    ["Frames compared", detail.compared + "%", "Share of the overlap where one frame is clearly sharper."],
  ];
  const regions = seam.uncertain_regions.length
    ? table(["Review area", "Position", "Size"], seam.uncertain_regions.slice(0, 8).map(region => [
      "Area", region.x + ", " + region.y, region.width + "×" + region.height,
    ]))
    : el("p", {className: "result-empty", text: "No sizeable area was left undecided."});
  pick("#join-seam-notes").replaceChildren(
    ...cards.map(([title, value, note]) => el("div", {className: "join-note"},
      el("strong", {text: title + " · " + value}), el("span", {text: note}),
    )),
    el("div", {className: "join-note full"}, el("strong", {text: "How the seam was chosen"}),
      el("span", {text: seam.notice})),
    el("div", {className: "join-note full"}, el("strong", {text: "Areas to review"}), regions),
  );
  if (detail.notices?.length) pick("#join-seam-notes").append(notices(detail.notices));
}

function renderJoinResult() {
  const result = joinState.result;
  pick("#join-review").hidden = !result;
  if (!result) return;
  pick("#join-notice").textContent = result.notice;
  pick("#join-candidates").hidden = false;
  renderJoinCandidates();
  renderJoinPreview();
  pick("#join-save").disabled = !chosenCandidate();
}

export async function enterJoin(app) {
  const state = app.state;
  joinState.app = app;
  if (!state.current) return;
  pick("#join-subtitle").textContent = state.current.title
    + " · choose the two overlapping captures of one face.";
  try {
    await loadFrames(app);
    showJoinStatus("");
  } catch (error) {
    showJoinStatus(error.message, true);
  }
}

export function wireJoin(app) {
  const {api, notify} = app;
  joinState.app = app;
  pick("#join-side").addEventListener("change", () => {
    if (joinState.result) {
      showJoinStatus("Side changed. Find the overlap again.");
      joinState.result = null;
      joinState.candidate = null;
      pick("#join-review").hidden = true;
    }
  });
  document.querySelectorAll("[data-join-view]").forEach(item => item.addEventListener("click", () => {
    joinState.view = item.dataset.joinView;
    renderJoinPreview();
  }));
  pick("#join-form").addEventListener("submit", async event => {
    event.preventDefault();
    const payload = joinPayload();
    if (!payload.first || !payload.second) {
      showJoinStatus("Choose both frames to join.", true);
      return;
    }
    if (payload.first === payload.second) {
      showJoinStatus("Choose two different photographs.", true);
      return;
    }
    showJoinStatus("Searching the shared area for the frame offset…");
    try {
      const result = await api("/api/stitches/preview", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      joinState.result = result;
      joinState.candidate = result.chosen ?? result.candidates[0] ?? null;
      joinState.view = "image";
      pick("#join-review").hidden = false;
      showJoinStatus(result.notice, result.ambiguous);
      renderJoinResult();
    } catch (error) {
      joinState.result = null;
      joinState.candidate = null;
      pick("#join-review").hidden = true;
      showJoinStatus(error.message, true);
    }
  });
  pick("#join-clear").addEventListener("click", () => {
    joinState.result = null;
    joinState.candidate = null;
    pick("#join-review").hidden = true;
    showJoinStatus("");
    pick("#join-form").reset();
  });
  pick("#join-save").addEventListener("click", async () => {
    const candidate = chosenCandidate();
    if (!candidate) return;
    const payload = joinPayload();
    const body = {
      first: payload.first, second: payload.second, side: payload.side,
      title: payload.title, material: payload.material, lot: payload.lot, surface: payload.surface,
      alignment: {dx: candidate.dx, dy: candidate.dy},
      seam: payload.seam,
    };
    pick("#join-save").disabled = true;
    showJoinStatus("Saving the joined working image at full resolution…");
    try {
      const record = await api("/api/stitches", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      notify("Joined working image saved");
      await app.openInspection(record.inspection_id, true);
    } catch (error) {
      showJoinStatus(error.message, true);
      pick("#join-save").disabled = false;
    }
  });
}
