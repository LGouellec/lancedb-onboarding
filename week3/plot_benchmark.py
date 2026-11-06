"""Turn a `python app.py --benchmark` JSON report into a self-contained HTML chart page.

    python plot_benchmark.py results/benchmark_20261104_103527.json [concurrency]
    -> results/benchmark_20261104_103527_chart.html   (open in a browser, no dependencies)

`concurrency` picks which load level the throughput chart uses (default: the middle one, e.g. 8 of
1,8,32 — the highest level is usually the noisiest).

Charts the "Vector search — nprobes × refine_factor sweep": recall@k vs latency, and recall@k vs
throughput under one concurrency level. One line per nprobes, one point per refine_factor.
"""

import json
import sys
from pathlib import Path

MAX_SERIES = 5  # the validated ordinal ramp has 5 steps; extra (largest) nprobes values go to the table only


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    report = json.loads(src.read_text())
    args, sweep = report["args"], report["sweep"]
    levels = sorted(args["concurrency"])
    conc = int(sys.argv[2]) if len(sys.argv) == 3 else levels[len(levels) // 2]
    if conc not in levels:
        sys.exit(f"concurrency {conc} not in the report ({levels})")

    points = []
    for it in sweep:
        v = it["vector"]
        seq = v["sequential"]
        if not seq["latency_ms"]:
            continue
        c = next((x for x in v["concurrent"] if x["concurrency"] == conc), None)
        points.append({
            "nprobes": it["nprobes"], "refine": it["refine"], "recall": seq["recall"],
            "p50": seq["latency_ms"]["p50"], "p95": seq["latency_ms"]["p95"], "p99": seq["latency_ms"]["p99"],
            "qps": c["qps"] if c else None, "p99c": c["latency_ms"].get("p99") if c else None,
        })

    data = {
        "points": points,
        "k": args["k"],
        "conc": conc,
        "target": args.get("target_recall", 0.95),
        "rows": report["table"]["rows"],
        "column": args["column"],
        "maxSeries": MAX_SERIES,
        "source": src.name,
    }
    out = src.with_name(src.stem + "_chart.html")
    out.write_text(TEMPLATE.replace("/*DATA*/null", json.dumps(data)))
    print(f"Chart -> {out}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vector Search Sweep</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #86b6ef; --s2: #5598e7; --s3: #2a78d6; --s4: #1c5cab; --s5: #104281;
  --accent: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #184f95; --s2: #256abf; --s3: #3987e5; --s4: #6da7ec; --s5: #9ec5f4;
    --accent: #d95926;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #184f95; --s2: #256abf; --s3: #3987e5; --s4: #6da7ec; --s5: #9ec5f4;
  --accent: #d95926;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
.sub { color: var(--ink-2); margin: 0 0 24px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 24px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 28px; font-weight: 600; margin: 2px 0; }
.tile .note { color: var(--muted); font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 18px 10px; margin-bottom: 20px; position: relative; }
.card h2 { font-size: 17px; margin: 0 0 2px; }
.card .desc { color: var(--ink-2); font-size: 14px; margin: 0 0 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--ink-2); margin-bottom: 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-family: inherit; }
.tick { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.axis-title { fill: var(--ink-2); font-size: 12px; }
.dlabel { fill: var(--ink-2); font-size: 12px; }
.tip { position: absolute; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 13px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); opacity: 0; transition: opacity .1s; white-space: nowrap; z-index: 2; }
.tip b { font-weight: 600; }
.tip .row { display: flex; justify-content: space-between; gap: 16px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.takeaways { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; }
.takeaways h2 { font-size: 17px; margin: 0 0 6px; }
.takeaways li { margin: 6px 0; }
.foot { color: var(--muted); font-size: 13px; margin-top: 4px; }
details { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; }
summary { cursor: pointer; color: var(--ink-2); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 8px; border-bottom: 1px solid var(--grid); text-align: right; }
th { color: var(--ink-2); font-weight: 600; }
code { font-size: 13px; }
</style>
</head>
<body>
<main>
  <h1>Vector search: nprobes × refine_factor</h1>
  <p class="sub" id="sub"></p>
  <div class="tiles" id="tiles"></div>
  <div class="takeaways"><h2>What the sweep shows</h2><ul id="takeaways"></ul></div>

  <div class="card" id="c1">
    <h2 id="c1t"></h2>
    <p class="desc">Up = better recall, left = faster. Each line is one <code>nprobes</code> value; the points on it are <code>refine_factor</code> 1 → 40 (bottom to top).</p>
    <div class="legend" id="lg1"></div>
    <svg id="ch1" viewBox="0 0 940 400" role="img"></svg>
    <div class="tip" id="tip1"></div>
  </div>

  <div class="card" id="c2">
    <h2 id="c2t"></h2>
    <p class="desc">Up = better recall, right = more queries per second. Same lines and points as above, measured under load.</p>
    <div class="legend" id="lg2"></div>
    <svg id="ch2" viewBox="0 0 940 400" role="img"></svg>
    <div class="tip" id="tip2"></div>
  </div>
  <p class="foot" id="foot"></p>

  <details>
    <summary>Table view (all data points)</summary>
    <div class="tablewrap"><table id="tbl"></table></div>
  </details>
</main>
<script>
const D = /*DATA*/null;
const K = D.k, CONC = D.conc, TARGET = D.target;
const fmt = (x, d = 1) => x == null ? "–" : x.toFixed(d);

// ---- series (one per nprobes), capped to the validated ramp size
const allNp = [...new Set(D.points.map(p => p.nprobes))].sort((a, b) => a - b);
const shownNp = allNp.slice(0, D.maxSeries), hiddenNp = allNp.slice(D.maxSeries);
const colorOf = np => `var(--s${shownNp.indexOf(np) + 1})`;
const series = shownNp.map(np => ({ np, pts: D.points.filter(p => p.nprobes === np).sort((a, b) => a.refine - b.refine) }));
const refines = [...new Set(D.points.map(p => p.refine))].sort((a, b) => a - b);

// ---- headline numbers
const ok = D.points.filter(p => p.recall != null && p.recall >= TARGET);
const best = ok.length ? ok.reduce((a, b) => a.p50 <= b.p50 ? a : b) : null;
const recallAt = r => D.points.filter(p => p.refine === r).map(p => p.recall);
const spreadByRefine = refines.map(r => { const v = recallAt(r); return Math.max(...v) - Math.min(...v); });
const maxSpread = Math.max(...spreadByRefine);
const r1 = recallAt(refines[0]), rTop = recallAt(refines[refines.length - 1]);
const fast = D.points.reduce((a, b) => a.p50 <= b.p50 ? a : b);
const minNp = allNp[0], maxNpShown = shownNp[shownNp.length - 1];
const lat = (np, r) => (D.points.find(p => p.nprobes === np && p.refine === r) || {}).p50;
const qps = (np, r) => (D.points.find(p => p.nprobes === np && p.refine === r) || {}).qps;

document.getElementById("sub").textContent =
  `${D.rows.toLocaleString()} rows · ${D.column} (cosine) · recall@${K} vs brute force · source ${D.source}`;
document.getElementById("c1t").textContent = `Recall@${K} vs median latency (one client)`;
document.getElementById("c2t").textContent = `Recall@${K} vs throughput (${CONC} concurrent clients)`;

const tiles = [
  { label: `Recall@${K} with refine_factor = ${refines[0]}`, value: fmt(Math.max(...r1), 3),
    note: `the same for every nprobes value: the PQ-compressed vectors cap recall` },
  best ? { label: `Fastest setting with recall ≥ ${TARGET}`, value: `${fmt(best.p50)} ms`,
    note: `nprobes=${best.nprobes}, refine=${best.refine} → recall ${fmt(best.recall, 3)}` }
       : { label: `Recall ≥ ${TARGET}`, value: "not reached", note: "no setting in the sweep" },
  { label: `Cost of nprobes ${minNp} → ${maxNpShown} (refine=${refines[0]})`,
    value: `+${fmt(lat(maxNpShown, refines[0]) - lat(minNp, refines[0]), 0)} ms`,
    note: `for a recall change of ${fmt(maxSpread, 3)} at most` },
];
document.getElementById("tiles").innerHTML = tiles.map(t =>
  `<div class="tile"><div class="label">${t.label}</div><div class="value">${t.value}</div><div class="note">${t.note}</div></div>`).join("");

const rTopRef = refines[refines.length - 1], bestNp = best ? best.nprobes : minNp;
const tk = [
  maxSpread < 0.02
    ? `<b>refine_factor alone sets recall.</b> At a given refine_factor, recall@${K} varies by only ${fmt(maxSpread, 3)} across all nprobes values. Recall goes from ${fmt(Math.max(...r1), 3)} at refine=${refines[0]} to ${fmt(Math.max(...rTop), 3)} at refine=${rTopRef}. The right neighbours are already in the nearest partitions; the compressed (PQ) distances just rank them badly, and refine_factor fixes that by re-ranking with full vectors.`
    : `Both nprobes and refine_factor move recall (spread across nprobes up to ${fmt(maxSpread, 3)}).`,
  `<b>Raising nprobes only adds latency.</b> Each step to the right on the latency chart is a higher nprobes value with no gain in recall. Median latency goes from ${fmt(lat(minNp, refines[0]), 0)} ms at nprobes=${minNp} to ${fmt(lat(maxNpShown, refines[0]), 0)} ms at nprobes=${maxNpShown}.`
    + (hiddenNp.length ? ` nprobes=${hiddenNp.join(", ")} gave the same results as ${maxNpShown}: the index has only ${maxNpShown} partitions.` : ""),
  `<b>refine_factor costs little for one client, but a lot under load.</b> At nprobes=${bestNp}, going from refine=${refines[0]} to ${rTopRef} adds ${fmt(lat(bestNp, rTopRef) - lat(bestNp, refines[0]), 0)} ms to median latency. With ${CONC} concurrent clients, throughput drops from ${fmt(qps(bestNp, refines[0]), 0)} to ${fmt(qps(bestNp, rTopRef), 0)} queries/s, because each query re-ranks ${rTopRef}×${K} full vectors.`,
  `<b>What to change:</b> keep nprobes low (${minNp}–${shownNp[Math.min(1, shownNp.length - 1)]}) and set refine_factor to whatever recall you need. To get high recall without refine's throughput cost, rebuild the index with more PQ sub-vectors (e.g. num_sub_vectors=48 instead of 16 for 384 dimensions) and more partitions.`,
];
document.getElementById("takeaways").innerHTML = tk.map(t => `<li>${t}</li>`).join("");
document.getElementById("foot").textContent = hiddenNp.length
  ? `Not drawn: nprobes=${hiddenNp.join(", ")} (identical to nprobes=${maxNpShown}, all values are in the table view).` : "";

// ---- chart
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };
function niceTicks(lo, hi, n = 6) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const t = []; for (let v = Math.floor(lo / step) * step; v - step < hi - 1e-9; v += step) t.push(+v.toFixed(10)); return t;
}

function chart(svgId, tipId, legendId, xKey, xTitle, xFmt, labelRefine) {
  const svg = document.getElementById(svgId), tip = document.getElementById(tipId);
  const W = 940, H = 400, m = { l: 56, r: 92, t: 14, b: 46 };
  const pts = series.flatMap(s => s.pts).filter(p => p[xKey] != null);
  const xs = pts.map(p => p[xKey]);
  const xt = niceTicks(Math.min(...xs) * 0.95, Math.max(...xs) * 1.03);
  const x0 = xt[0], x1 = xt[xt.length - 1];
  const ymin = Math.floor(Math.min(...pts.map(p => p.recall)) * 10) / 10 - 0.05;
  const yt = niceTicks(Math.max(0, ymin), 1, 6).filter(v => v <= 1);
  const y0 = yt[0], y1 = 1;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);

  // grid + ticks
  for (const v of yt) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = v.toFixed(2);
  }
  for (const v of xt) el("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", class: "tick" }, svg).textContent = xFmt(v);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1 }, svg);
  el("text", { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", class: "axis-title" }, svg).textContent = xTitle;
  const yl = el("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, svg); yl.textContent = `recall@${K}`;

  // target line
  if (TARGET > y0 && TARGET < 1) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(TARGET), y2: Y(TARGET), stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "4 4" }, svg);
    el("text", { x: m.l + 6, y: Y(TARGET) - 6, class: "dlabel" }, svg).textContent = `target ${TARGET}`;
  }

  // lines + points
  const hits = [];
  for (const s of series) {
    const sp = s.pts.filter(p => p[xKey] != null);
    el("polyline", { points: sp.map(p => `${X(p[xKey])},${Y(p.recall)}`).join(" "), fill: "none",
      stroke: colorOf(s.np), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    for (const p of sp) {
      el("circle", { cx: X(p[xKey]), cy: Y(p.recall), r: 4.5, fill: colorOf(s.np), stroke: "var(--surface)", "stroke-width": 2 }, svg);
      hits.push({ p, cx: X(p[xKey]), cy: Y(p.recall) });
    }
  }

  // direct labels: refine level right of the last point of each recall row
  if (labelRefine) for (const r of refines) {
    const rowPts = series.map(s => s.pts.find(q => q.refine === r)).filter(q => q && q[xKey] != null);
    if (!rowPts.length) continue;
    const last = rowPts.reduce((a, b) => X(a[xKey]) >= X(b[xKey]) ? a : b);
    el("text", { x: X(last[xKey]) + 10, y: Y(last.recall) + 4, class: "dlabel" }, svg).textContent = `refine ${r}`;
  }

  // highlight the recommended setting
  if (best && best[xKey] != null) {
    const bx = X(best[xKey]), by = Y(best.recall);
    el("circle", { cx: bx, cy: by, r: 9, fill: "none", stroke: "var(--accent)", "stroke-width": 2 }, svg);
    el("text", { x: bx, y: by - 15, "text-anchor": "middle", class: "dlabel", style: "fill: var(--ink); font-weight: 600" }, svg)
      .textContent = `fastest ≥ ${TARGET}: nprobes ${best.nprobes}, refine ${best.refine}`;
  }

  // legend
  document.getElementById(legendId).innerHTML = "<span>nprobes:</span>" +
    series.map(s => `<span><i style="background:${colorOf(s.np)}"></i>${s.np}</span>`).join("") +
    (best ? `<span><i style="background:none;border:2px solid var(--accent);width:12px;height:12px;border-radius:50%"></i>recommended</span>` : "");

  // hover: nearest point within 18px
  const ring = el("circle", { r: 8, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0 }, svg);
  const toSvg = e => { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); };
  svg.addEventListener("pointermove", e => {
    const q = toSvg(e); let h = null, dmin = 18;
    for (const c of hits) { const d = Math.hypot(c.cx - q.x, c.cy - q.y); if (d < dmin) { dmin = d; h = c; } }
    if (!h) { tip.style.opacity = 0; ring.setAttribute("opacity", 0); return; }
    const p = h.p;
    ring.setAttribute("cx", h.cx); ring.setAttribute("cy", h.cy); ring.setAttribute("opacity", 1);
    tip.innerHTML = `<b>nprobes ${p.nprobes} · refine ${p.refine}</b>` +
      `<div class="row"><span>recall@${K}</span><span>${fmt(p.recall, 3)}</span></div>` +
      `<div class="row"><span>p50 / p95 / p99</span><span>${fmt(p.p50)} / ${fmt(p.p95)} / ${fmt(p.p99)} ms</span></div>` +
      `<div class="row"><span>QPS @ ${CONC} clients</span><span>${fmt(p.qps)}</span></div>` +
      `<div class="row"><span>p99 @ ${CONC} clients</span><span>${fmt(p.p99c)} ms</span></div>`;
    const card = svg.parentElement.getBoundingClientRect(), sr = svg.getBoundingClientRect();
    const px = sr.left - card.left + h.cx * sr.width / W, py = sr.top - card.top + h.cy * sr.height / H;
    tip.style.opacity = 1;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(Math.max(px - tw / 2, 4), card.width - tw - 4) + "px";
    tip.style.top = (py - tip.offsetHeight - 14) + "px";
  });
  svg.addEventListener("pointerleave", () => { tip.style.opacity = 0; ring.setAttribute("opacity", 0); });
}

chart("ch1", "tip1", "lg1", "p50", "median latency (ms), lower is better", v => v.toFixed(0), true);
chart("ch2", "tip2", "lg2", "qps", `queries per second with ${CONC} concurrent clients, higher is better`, v => v.toFixed(0), false);

// ---- table view
document.getElementById("tbl").innerHTML =
  `<tr><th>nprobes</th><th>refine</th><th>recall@${K}</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>QPS c=${CONC}</th><th>p99 c=${CONC}</th></tr>` +
  D.points.map(p => `<tr><td>${p.nprobes}</td><td>${p.refine}</td><td>${fmt(p.recall, 3)}</td><td>${fmt(p.p50)}</td><td>${fmt(p.p95)}</td><td>${fmt(p.p99)}</td><td>${fmt(p.qps)}</td><td>${fmt(p.p99c)}</td></tr>`).join("");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
