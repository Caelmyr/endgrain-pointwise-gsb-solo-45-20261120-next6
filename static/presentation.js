const number = new Intl.NumberFormat("en", {maximumFractionDigits: 2});
const detail = new Intl.NumberFormat("en", {maximumFractionDigits: 3});
const time = new Intl.DateTimeFormat("en", {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});

export function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  Object.entries(options).forEach(([key, value]) => {
    if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "text") node.textContent = value;
    else if (key === "onclick") node.addEventListener("click", value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  });
  node.append(...children);
  return node;
}

export function displayValue(value, mode, unit = "px", metrics = {}) {
  if (value === null || value === undefined) return "—";
  if (mode === "bool") return value ? "Yes" : "No";
  if (mode === "%") return number.format(value * 100) + "%";
  if (mode === "area") return number.format(value) + " " + unit + "²";
  if (mode === "linear") return number.format(value) + " " + unit;
  if (mode === "density") return number.format(value) + " / " + metrics.density_basis;
  if (mode === "°") return number.format(value) + "°";
  if (mode === "px") return number.format(value) + " px";
  return detail.format(value);
}

export function shortTime(value) {
  return time.format(new Date(value));
}

export function notices(entries) {
  return el("ul", {className: "analysis-notices"}, ...entries.map(text => el("li", {text})));
}

export function button(text, onclick, className = "button outline") {
  return el("button", {type: "button", text, className, onclick});
}

export function link(text, href, className = "button outline") {
  return el("a", {text, href, className});
}

export function createToolInput(spec, current) {
  const fields = {"data-setting": spec.key, name: spec.key};
  const defaults = current ?? spec.default;
  if (spec.values.length) {
    const select = el("select", fields, ...spec.values.map(value => el("option", {value, text: value.replaceAll("_", " ")})));
    select.value = defaults;
    return el("label", {text: spec.title}, select);
  }
  if (typeof spec.default === "boolean") {
    const input = el("input", {...fields, type: "checkbox", checked: defaults});
    return el("label", {className: "checkbox full"}, input, document.createTextNode(spec.title));
  }
  return el("label", {text: spec.title}, el("input", {...fields,
    type: "number", value: defaults, step: "any", inputMode: "decimal", required: true,
  }));
}

function lines(ctx, xs, ys, xf, yf, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  let fresh = true;
  xs.forEach((x, index) => {
    if (ys[index] === null) {
      fresh = true;
    } else {
      if (fresh) ctx.moveTo(xf(x), yf(ys[index]));
      else ctx.lineTo(xf(x), yf(ys[index]));
      fresh = false;
    }
  });
  ctx.stroke();
}

export function plot(specification) {
  const graph = el("canvas", {width: 600, height: 184, className: "plot",
    role: "img", "aria-label": specification.x_label + " versus " + specification.y_label,
  });
  const ctx = graph.getContext("2d");
  const left = 42, right = 580, top = 12, bottom = 145;
  let xmin = 0, xmax = 1, ymin = 0, ymax = 1;
  if (specification.kind === "histogram") {
    xmax = 31;
    ymax = Math.max(1, ...Object.values(specification.channels).flat());
  } else if (specification.kind === "profile" && specification.x.length) {
    const valid = specification.y.filter(y => y !== null);
    xmin = Math.min(...specification.x);
    xmax = Math.max(...specification.x);
    ymin = valid.length ? Math.min(...valid) - 0.01 : 0;
    ymax = valid.length ? Math.max(...valid) + 0.01 : 1;
  } else if (specification.kind === "bars" && specification.rows.length) {
    xmin = specification.rows[0].x;
    xmax = specification.rows.at(-1).x;
    ymax = Math.max(1e-5, ...specification.rows.map(row => row.value));
  }
  const xf = value => left + (value - xmin) * (right - left) / Math.max(xmax - xmin, 0.000001);
  const yf = value => bottom - (value - ymin) * (bottom - top) / Math.max(ymax - ymin, 0.000001);
  ctx.strokeStyle = "#d7daca";
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui";
  ctx.fillStyle = "#676c5e";
  for (let i = 0; i < 4; i++) {
    const value = ymin + (ymax - ymin) * i / 3;
    const y = yf(value);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.fillText(detail.format(value), 2, y + 4, 37);
  }
  ctx.textAlign = "center";
  ctx.fillText(specification.x_label, (left + right) / 2, 174, 425);
  ctx.textAlign = "left";
  ctx.fillText(detail.format(xmin), left, 160);
  ctx.textAlign = "right";
  ctx.fillText(detail.format(xmax), right, 160);
  if (specification.kind === "profile") {
    ctx.strokeStyle = "#b98048";
    ctx.lineWidth = 1;
    (specification.markers || []).forEach(mark => {
      const x = xf(mark);
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    });
    lines(ctx, specification.x, specification.y, xf, yf, "#465438");
  } else if (specification.kind === "bars") {
    const width = Math.max(2, Math.min(35, (right - left) / Math.max(specification.rows.length, 1) * 0.78));
    ctx.fillStyle = "#728565";
    specification.rows.forEach(row => ctx.fillRect(xf(row.x) - width / 2, yf(row.value), width, bottom - yf(row.value)));
  } else {
    Object.entries(specification.channels).forEach(([channel, values]) => {
      const colors = {r: "#b2684d", g: "#698155", b: "#5d759b"};
      lines(ctx, values.map((_, ix) => ix), values, xf, yf, colors[channel]);
    });
  }
  return graph;
}

export function metricCards(tool, result) {
  const points = (tool?.metrics && Object.entries(tool.metrics)) || [];
  return el("div", {className: "metrics"}, ...points.map(([key, [label, mode]]) => el("div", {className: "metric"},
    el("span", {text: label}),
    el("strong", {text: displayValue(result.metrics[key], mode, result.unit, result.metrics)}),
  )));
}

const PORE_DECISIONS = {
  counted_inside: "Whole inside region",
  counted_closed: "Closed across the line",
  counted_partial: "Partial, capture edge",
  outside_region: "Visible, centre outside region",
  excluded_at_capture: "Not counted, capture edge",
  excluded_too_small: "Not counted, too small",
  excluded_too_big: "Not counted, too large",
  excluded_shape: "Not counted, shape",
};

export function decisionLabel(value) {
  return PORE_DECISIONS[value] || value || "—";
}

export function objectTable(run) {
  const measures = run.result;
  let rows = [];
  let headers = [];
  let leading = null;
  if (run.kind === "rings") {
    headers = ["Crossing", "Position", "Previous interval", "Origin"];
    rows = (measures.crossings || []).map((entry, ix) => [
      ix + 1, displayValue(entry.distance, "linear", measures.unit),
      displayValue(entry.gap_from_previous, "linear", measures.unit), entry.origin,
    ]);
  } else if (run.kind === "cracks") {
    headers = ["Trace", "Length", "Width", "Junctions", "Edge"];
    rows = (measures.objects || []).map(entry => [
      entry.id, displayValue(entry.length, "linear", measures.unit),
      displayValue(entry.median_width, "linear", measures.unit), entry.junctions, entry.touches_support_edge ? "Yes" : "No",
    ]);
  } else if (run.kind === "pores") {
    const basis = measures.basis || {};
    const notes = (basis.contours || []).map(line => el("li", {text: line}));
    if (notes.length) {
      leading = el("div", {className: "basis-note"}, el("strong", {text: "Measured-area basis"}), el("ul", {}, ...notes));
    }
    headers = ["Pore", "Counted as", "Full contour area", "Area inside region", "Diameter", "Roundness"];
    rows = (measures.objects || []).map(entry => [
      entry.id ?? "—",
      decisionLabel(entry.decision),
      displayValue(entry.area, "area", measures.unit),
      displayValue(entry.measured_area, "area", measures.unit),
      displayValue(entry.diameter, "linear", measures.unit), displayValue(entry.roundness),
    ]);
  } else if (run.kind === "colors") {
    headers = ["Swatch", "Qualified pixels", "Coverage"];
    rows = (measures.objects || []).map(entry => [entry.label, entry.pixels, displayValue(entry.coverage, "%")]);
  }
  if (!rows.length) return leading;
  const detail = el("details", {className: "manual"}, el("summary", {text: "Show readings · " + rows.length}),
    table(headers, rows),
  );
  return leading ? el("div", {}, leading, detail) : detail;
}

export function table(headers, rows) {
  const heading = el("thead", {},
    el("tr", {}, ...headers.map(value => el("th", {text: value, scope: "col"}))),
  );
  const body = el("tbody", {}, ...rows.map(row => el("tr", {}, ...row.map(value => el("td", {text: value ?? "—"})))));
  return el("div", {className: "table-wrap"}, el("table", {}, heading, body));
}
