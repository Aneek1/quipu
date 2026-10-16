// Charts and figures shared by the pages. Each block only runs on a page that has its element.
const NS = "http://www.w3.org/2000/svg";
const MONO = "Geist Mono, JetBrains Mono, ui-monospace, monospace";

// Code-share data, from the build manifest.
const LANGS = [
  ["Java", 34.69], ["JavaScript", 17.19], ["Python", 14.51], ["PHP", 12.38],
  ["HTML (capped)", 10.00], ["Go", 4.24], ["TypeScript", 2.78], ["CSS", 2.76],
  ["Shell", 0.84], ["SQL", 0.53], ["Dockerfile", 0.08]
];

function barRow(parent, name, width, value, acc) {
  const row = document.createElement("div"); row.className = "lang";
  const label = document.createElement("span"); label.textContent = name;
  const track = document.createElement("span"); track.className = "track";
  const fill = document.createElement("span"); fill.className = "fill" + (acc ? " acc" : "");
  fill.style.width = width.toFixed(1) + "%";
  track.appendChild(fill);
  const pct = document.createElement("span"); pct.className = "pct"; pct.textContent = value;
  row.append(label, track, pct);
  parent.appendChild(row);
}

const langs = document.getElementById("langs");
if (langs) {
  const max = LANGS[0][1];
  LANGS.forEach(([name, pct], i) => barRow(langs, name, Math.max(pct / max * 100, 0.8), pct.toFixed(pct < 1 ? 2 : 1) + "%", i === 0));
}

// Quipu: a main cord with pendant cords; knots encode the rounded share in tens and units.
const quipuSvg = document.getElementById("quipu");
if (quipuSvg) {
  const el = (tag, attrs) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); quipuSvg.appendChild(n); return n; };
  const cordShades = ["#71717a", "#52525b", "#a1a1aa", "#52525b", "#71717a", "#52525b", "#a1a1aa", "#52525b"];
  const cords = LANGS.slice(0, 8);
  const x0 = 38, gap = (440 - 2 * x0) / (cords.length - 1), cordTop = 34;
  // faint guide lines for the tens and units registers
  el("line", { x1: 14, x2: 426, y1: 140, y2: 140, stroke: "#27272a", "stroke-width": 1, "stroke-dasharray": "2 4" });
  el("path", { d: `M10 ${cordTop} Q220 ${cordTop - 16} 430 ${cordTop}`, stroke: "#e4e4e7", "stroke-width": 3.5, fill: "none", "stroke-linecap": "round" });
  cords.forEach(([name, pct], i) => {
    const x = x0 + i * gap, v = Math.round(pct), tens = Math.floor(v / 10), units = v % 10;
    const hi = i === 0;
    const cord = hi ? "#f0a646" : cordShades[i];
    const knot = hi ? "#f0a646" : "#d4d4d8";
    const sway = (i % 2 ? 1 : -1) * 3;
    el("path", { d: `M${x} ${cordTop - 2} C${x + sway} 110 ${x - sway} 200 ${x} 262`, stroke: cord, "stroke-width": hi ? 2.4 : 1.8, fill: "none", "stroke-linecap": "round" });
    for (let k = 0; k < tens; k++) el("circle", { cx: x, cy: 78 + k * 11, r: 4.6, fill: knot });
    for (let k = 0; k < units; k++) el("ellipse", { cx: x, cy: 152 + k * 10, rx: 3.8, ry: 3, fill: knot, opacity: hi ? 1 : .8 });
    const t = el("text", { x, y: 286, "text-anchor": "middle", "font-size": 10.5, fill: hi ? "#f0a646" : "#a1a1aa", "font-family": MONO });
    t.textContent = name.replace(" (capped)", "").replace("JavaScript", "JS").replace("TypeScript", "TS");
  });
}

// Bits per byte at step 8,129, from the quipu-moe-1B-A149M model card.
const bpbEl = document.getElementById("bpb");
if (bpbEl) {
  const BPB = [["Code", 0.448], ["Tamil", 0.491], ["Hindi (Devanagari)", 0.568], ["English", 0.959],
    ["Indonesian", 1.157], ["Malay", 1.162], ["Korean", 1.199], ["Japanese", 1.234],
    ["Hindi (romanised)", 1.255], ["Urdu (romanised)", 1.582], ["Chinese (Trad.)", 1.728], ["Chinese (Simp.)", 1.768]];
  BPB.forEach(([name, v]) => barRow(bpbEl, name, v / 2 * 100, v.toFixed(2), name === "Code"));
}

// Validation loss every 100 steps, from results/runs/quipu-114m-weekend.json.
const curveSvg = document.getElementById("curve");
if (curveSvg) {
  const VAL = [[100, 6.378], [200, 5.471], [300, 4.926], [400, 4.612], [500, 4.364], [600, 4.17], [700, 4.047], [800, 3.959], [900, 3.879], [1000, 3.819], [1100, 3.774], [1200, 3.744], [1300, 3.708], [1400, 3.676], [1500, 3.648], [1600, 3.63], [1700, 3.601], [1800, 3.579], [1900, 3.567], [2000, 3.55], [2100, 3.529], [2200, 3.514], [2300, 3.509], [2400, 3.485], [2500, 3.472], [2600, 3.468], [2700, 3.452], [2800, 3.441], [2900, 3.441], [3000, 3.424], [3100, 3.41], [3200, 3.402], [3300, 3.393], [3400, 3.385], [3500, 3.376], [3600, 3.369], [3700, 3.362], [3800, 3.353], [3900, 3.348], [4000, 3.346], [4100, 3.332], [4200, 3.326], [4300, 3.323], [4400, 3.317], [4500, 3.312], [4600, 3.308], [4700, 3.303], [4800, 3.299], [4900, 3.293], [5000, 3.29], [5100, 3.286], [5200, 3.281], [5300, 3.282], [5400, 3.281], [5500, 3.276], [5600, 3.274], [5700, 3.275]];
  const svg = curveSvg;
  const W = 640, H = 300, L = 44, R = 14, T = 14, B = 34;
  const xmax = 5722, ymin = 3.0, ymax = 6.6;
  const X = s => L + (s / xmax) * (W - L - R);
  const Y = v => T + (ymax - v) / (ymax - ymin) * (H - T - B);
  const add = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text) n.textContent = text; svg.appendChild(n); return n; };
  const defs = add("defs", {});
  const g = document.createElementNS(NS, "linearGradient");
  g.setAttribute("id", "fade"); g.setAttribute("x1", "0"); g.setAttribute("x2", "0"); g.setAttribute("y1", "0"); g.setAttribute("y2", "1");
  for (const [o, a] of [["0", .22], ["1", 0]]) { const s = document.createElementNS(NS, "stop"); s.setAttribute("offset", o); s.setAttribute("stop-color", "#f0a646"); s.setAttribute("stop-opacity", a); g.appendChild(s); }
  defs.appendChild(g);
  for (const v of [3, 4, 5, 6]) {
    add("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "#27272a", "stroke-width": 1 });
    add("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, fill: "#71717a", "font-family": MONO }, v.toFixed(1));
  }
  for (const s of [0, 1000, 2000, 3000, 4000, 5000]) {
    add("text", { x: X(s), y: H - 12, "text-anchor": "middle", "font-size": 11, fill: "#71717a", "font-family": MONO }, s === 0 ? "0" : (s / 1000) + "k");
  }
  const d = VAL.map(([s, v], i) => (i ? "L" : "M") + X(s).toFixed(1) + " " + Y(v).toFixed(1)).join(" ");
  const area = d + ` L ${X(VAL[VAL.length - 1][0]).toFixed(1)} ${Y(ymin)} L ${X(VAL[0][0]).toFixed(1)} ${Y(ymin)} Z`;
  add("path", { d: area, fill: "url(#fade)" });
  add("path", { d, stroke: "#f0a646", "stroke-width": 2, fill: "none", "stroke-linejoin": "round" });
  const last = VAL[VAL.length - 1];
  add("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, fill: "#09090b", stroke: "#f0a646", "stroke-width": 2 });
  add("text", { x: X(last[0]) - 8, y: Y(last[1]) - 12, "text-anchor": "end", "font-size": 12, fill: "#fafafa", "font-family": MONO }, last[1].toFixed(2));
}

// Greedy samples for one prompt, from results/milestones/samples.md (trimmed).
const stepper = document.getElementById("stepper"), sample = document.getElementById("sample");
if (stepper && sample) {
  const PROMPT = "Photosynthesis is the process by which";
  const SAMPLES = [
    [100, 6.40, "the first is the first of the first time of the first time of the first time of the first time…"],
    [250, 5.20, "the system is not the same. The current is the most common type of the system, which is the most common type of the system."],
    [500, 4.40, "the temperature is measured. The temperature of the temperature is measured by the temperature of the temperature…"],
    [1000, 3.85, "the body absorbs energy from the sun. The sun is the most abundant source of energy in the sun."],
    [2000, 3.58, "plants grow and reproduce. Plants are able to produce energy from the sun and sunlight."],
    [4000, 3.38, "plants convert carbon dioxide into oxygen and convert it into energy."],
    [5722, 3.31, "plants convert carbon dioxide into oxygen and release carbon dioxide into the atmosphere."]
  ];
  function show(i) {
    const [step, loss, text] = SAMPLES[i];
    [...stepper.children].forEach((b, j) => b.setAttribute("aria-selected", String(i === j)));
    sample.innerHTML = "";
    const b = document.createElement("b"); b.textContent = PROMPT + " ";
    const l = document.createElement("span"); l.className = "loss";
    l.textContent = `step ${step.toLocaleString()} · ${(step * 524288 / 1e9).toFixed(2)}B tokens seen · text val loss ${loss.toFixed(2)}`;
    sample.append(b, document.createTextNode(text), l);
  }
  SAMPLES.forEach(([step], i) => {
    const btn = document.createElement("button");
    btn.type = "button"; btn.setAttribute("role", "tab"); btn.textContent = step.toLocaleString();
    btn.addEventListener("click", () => show(i));
    stepper.appendChild(btn);
  });
  show(SAMPLES.length - 1);
}

// Decision tree: show or hide the options that were not chosen.
const toggle = document.getElementById("toggle-rejected"), tree = document.getElementById("tree");
if (toggle && tree) {
  toggle.addEventListener("click", () => {
    const hidden = tree.classList.toggle("hide-rejected");
    toggle.textContent = hidden ? "Show the rejected options" : "Show only what was chosen";
    toggle.setAttribute("aria-pressed", String(hidden));
  });
}
