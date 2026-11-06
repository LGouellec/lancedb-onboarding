import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lancedb.table import IvfPq
from lancedb.query import MatchQuery

db = lancedb.connect(
    "db://bench",
    host_override="http://localhost:10024",
    api_key="b583083048a24627a7a3e6544bab26c337e0980f5d02736bdf2ab34c6f7754af")

print(db.list_tables().tables)

table = db.open_table("fineweb_edu")

model_name = "sentence-transformers/all-MiniLM-L6-v2"

def embed_text(text):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(text)


BENCH_DIR = Path(__file__).parent / "bench_data"
RESULTS_DIR = Path(__file__).parent / "results"
STOPWORDS = set("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he her
here hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our
out over own same she should so some such than that the their them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours one two may many much
used use like make made well even first new get also however include including""".split())


def log(*a):
    print(*a, flush=True)


def keywords(text, n=3):
    """The `n` most frequent non-stopword words (>= 5 chars) of a document, used as its FTS query."""
    words = [w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in STOPWORDS]
    return " ".join(w for w, _ in Counter(words).most_common(n)) or "education"


def bench_prepare(args):
    """Sample query rows (embedding + keywords). In-distribution queries, so no model is needed."""
    path = BENCH_DIR / f"queries_{args.column}_n{args.num_queries}_s{args.seed}.parquet"
    if not path.exists():
        n_rows = table.count_rows()
        offsets = sorted(random.Random(args.seed).sample(range(n_rows), args.num_queries))
        t0 = time.time()
        rows = table.take_offsets(offsets).select(["id", "text", args.column]).to_arrow()
        BENCH_DIR.mkdir(exist_ok=True)
        pq.write_table(pa.table({
            "id": rows["id"],
            "terms": [keywords(t) for t in rows["text"].to_pylist()],
            "vector": rows[args.column],
        }), path)
        log(f"[prepare] sampled {args.num_queries} queries out of {n_rows:,} rows in {time.time() - t0:.0f}s")
    else:
        log(f"[prepare] using cached {path.name}")
    t = pq.read_table(path)
    return [{"id": i, "terms": s, "vector": np.asarray(v, dtype=np.float32)}
            for i, s, v in zip(t["id"].to_pylist(), t["terms"].to_pylist(), t["vector"].to_pylist())]


def bench_truth(args, queries, where):
    """Exact top-k (brute force, ~10s/query on 5M rows). Checkpointed after every query."""
    tag = re.sub(r"\W+", "_", where).strip("_") if where else "nofilter"
    path = BENCH_DIR / f"truth_{args.column}_k{args.k}_{tag}.json"
    truth = json.loads(path.read_text()) if path.exists() else {}
    todo = [q for q in queries[: args.num_truth] if q["id"] not in truth]
    log(f"[truth] {where or 'no filter'}: {len(truth)} cached, {len(todo)} to compute")
    for i, q in enumerate(todo, 1):
        t0 = time.time()
        qb = (table.search(q["vector"], vector_column_name=args.column).distance_type("cosine")
              .bypass_vector_index().select(["id"]).limit(args.k))
        if where:
            qb = qb.where(where, prefilter=True)
        truth[q["id"]] = qb.to_arrow()["id"].to_pylist()
        path.write_text(json.dumps(truth))
        log(f"  [{i}/{len(todo)}] {time.time() - t0:.1f}s")
    return truth


def vector_query(args, nprobes, refine, where=None):
    def run(q):
        qb = table.search(q["vector"], vector_column_name=args.column).distance_type("cosine").nprobes(nprobes)
        if refine > 1:
            qb = qb.refine_factor(refine)
        if where:
            qb = qb.where(where, prefilter=True)
        return qb.select(args.select).limit(args.k).to_arrow()
    return run


def baseline_queries(args):
    return {
        "fts": lambda q: table.search(q["terms"], query_type="fts", fts_columns="text")
            .select(args.select).limit(args.k).to_arrow(),
        "hybrid": lambda q: table.search(query_type="hybrid", vector_column_name=args.column)
            .vector(q["vector"]).text(q["terms"]).select(args.select).limit(args.k).to_arrow(),
        "filter_scan": lambda q: table.search().where(args.filter).select(args.select).limit(args.k).to_arrow(),
    }


def pct(lat_ms):
    if not lat_ms:
        return {}
    a = np.asarray(lat_ms)
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def measure_sequential(fn, queries, truth, k):
    lat, recalls, errors = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        try:
            res = fn(q)
        except Exception as e:
            errors.append(str(e)[:200])
            continue
        lat.append((time.perf_counter() - t0) * 1000)
        if truth and q["id"] in truth:
            gt = set(truth[q["id"]][:k])
            if gt:
                recalls.append(len(gt & set(res["id"].to_pylist())) / len(gt))
    return {"latency_ms": pct(lat), "recall": statistics.mean(recalls) if recalls else None,
            "errors": len(errors), "first_error": errors[0] if errors else None}


def measure_concurrent(fn, queries, concurrency, duration):
    deadline = time.perf_counter() + duration

    def worker(wid):
        lat, errs, i = [], 0, wid
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            try:
                fn(queries[i % len(queries)])
                lat.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errs += 1
            i += concurrency
        return lat, errs

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        out = list(pool.map(worker, range(concurrency)))
    lat = [x for r, _ in out for x in r]
    return {"concurrency": concurrency, "qps": len(lat) / (time.perf_counter() - t0),
            "errors": sum(e for _, e in out), "latency_ms": pct(lat)}


def measure(name, fn, queries, truth, args):
    for q in queries[: args.warmup]:
        try:
            fn(q)
        except Exception:
            pass
    seq = measure_sequential(fn, queries, truth, args.k)
    res = {"scenario": name, "sequential": seq, "concurrent": []}
    l = seq["latency_ms"]
    if not l:
        log(f"  {name}: FAILED ({seq['first_error']})")
        return res
    rec = f" recall@{args.k}={seq['recall']:.3f}" if seq["recall"] is not None else ""
    log(f"  {name}: p50={l['p50']:.1f}ms p95={l['p95']:.1f}ms p99={l['p99']:.1f}ms{rec}")
    for c in args.concurrency:
        r = measure_concurrent(fn, queries, c, args.duration)
        res["concurrent"].append(r)
        log(f"    c={c:<3} qps={r['qps']:.1f} p99={r['latency_ms'].get('p99', 0):.1f}ms errors={r['errors']}")
    return res


def table_info():
    info = {"rows": table.count_rows(), "version": table.version, "indices": []}
    for idx in table.list_indices():
        e = {"name": idx.name, "type": str(idx.index_type), "columns": list(idx.columns)}
        try:
            st = table.index_stats(idx.name)
            e.update(indexed=st.num_indexed_rows, unindexed=st.num_unindexed_rows,
                     partitions=getattr(st, "num_partitions", None))
        except Exception:
            pass
        info["indices"].append(e)
    return info


def f(x, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x)


def write_markdown(path, args, info, sweep, baselines, started):
    cs = args.concurrency
    conc_hdr = " | ".join(f"QPS c={c} | p99 c={c}" for c in cs)
    conc_sep = " | ".join("---: | ---:" for _ in cs)

    def conc_cells(r):
        by_c = {x["concurrency"]: x for x in r["concurrent"]}
        return " | ".join(f"{f(by_c[c]['qps']) if c in by_c else '-'} | "
                          f"{f(by_c[c]['latency_ms'].get('p99')) if c in by_c else '-'}" for c in cs)

    def row(r, prefix):
        s, l = r["sequential"], r["sequential"]["latency_ms"]
        err = s["errors"] + sum(x["errors"] for x in r["concurrent"])
        return (f"| {prefix} | {f(s['recall'], '{:.3f}')} | {f(l.get('p50'))} | {f(l.get('p95'))} | "
                f"{f(l.get('p99'))} | {conc_cells(r)} | {err} |")

    md = [f"# fineweb_edu benchmark — {started:%Y-%m-%d %H:%M}", "",
          f"- Rows: **{info['rows']:,}** (table version {info['version']})",
          f"- Vector column: `{args.column}` (cosine), k={args.k}, returned columns: `{','.join(args.select)}`",
          f"- Queries: {args.num_queries} rows sampled from the table (seed {args.seed}); "
          f"recall measured on {args.num_truth} of them vs brute force",
          f"- Filter: `{args.filter}` (prefilter) · concurrency levels {cs}, {args.duration:.0f}s each",
          "", "## Indices", "", "| name | type | columns | indexed | unindexed |", "|---|---|---|---:|---:|"]
    md += [f"| {i['name']} | {i['type']} | {', '.join(i['columns'])} | {i.get('indexed', '-')} | "
           f"{i.get('unindexed', '-')} |" for i in info["indices"]]
    for kind, title in [("vector", "Vector search"), ("vector_filter", f"Vector search + filter `{args.filter}`")]:
        md += ["", f"## {title} — nprobes × refine_factor sweep", "",
               f"| nprobes | refine | recall@{args.k} | p50 ms | p95 ms | p99 ms | {conc_hdr} | errors |",
               f"|---:|---:|---:|---:|---:|---:| {conc_sep} | ---: |"]
        md += [row(it[kind], f"{it['nprobes']} | {it['refine']}") for it in sweep]
        ok = [it for it in sweep if it[kind]["sequential"]["recall"] is not None
              and it[kind]["sequential"]["recall"] >= args.target_recall and it[kind]["sequential"]["latency_ms"]]
        if ok:
            best = min(ok, key=lambda it: it[kind]["sequential"]["latency_ms"]["p50"])
            md += ["", f"Fastest config with recall@{args.k} ≥ {args.target_recall}: **nprobes={best['nprobes']}, "
                       f"refine={best['refine']}** (p50 {best[kind]['sequential']['latency_ms']['p50']:.1f} ms, "
                       f"recall {best[kind]['sequential']['recall']:.3f})"]
        elif sweep:
            md += ["", f"No config reached recall@{args.k} ≥ {args.target_recall}."]
    if baselines:
        md += ["", "## Other query types", "",
               f"| scenario | recall | p50 ms | p95 ms | p99 ms | {conc_hdr} | errors |",
               f"|---|---:|---:|---:|---:| {conc_sep} | ---: |"]
        md += [row(r, r["scenario"]) for r in baselines]
        failed = [r for r in baselines if r["sequential"]["first_error"] and not r["sequential"]["latency_ms"]]
        md += [f"\n- `{r['scenario']}` failed: `{r['sequential']['first_error']}`" for r in failed]
    path.write_text("\n".join(md) + "\n")


def benchmark(argv):
    csv_int = lambda s: [int(x) for x in s.split(",")]
    p = argparse.ArgumentParser(prog="app.py --benchmark")
    p.add_argument("--column", default="text_embedding", help="text_embedding | embedding_mxbai")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--nprobes", type=csv_int, default=[5, 10, 20, 40, 64])
    p.add_argument("--refine", type=csv_int, default=[1, 5, 10])
    p.add_argument("--concurrency", type=csv_int, default=[1, 8, 32])
    p.add_argument("--duration", type=float, default=10, help="seconds per concurrency level")
    p.add_argument("--filter", default="int_score >= 4")
    p.add_argument("--select", type=lambda s: s.split(","), default=["id"], help="columns returned per hit, e.g. id,url,text to include payload cost")
    p.add_argument("--num-queries", type=int, default=200)
    p.add_argument("--num-truth", type=int, default=50, help="queries with brute-force ground truth")
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-baselines", action="store_true", help="skip fts / hybrid / filter-only queries")
    args = p.parse_args(argv)

    started = datetime.now()
    RESULTS_DIR.mkdir(exist_ok=True)
    md_path = RESULTS_DIR / f"benchmark_{started:%Y%m%d_%H%M%S}.md"
    json_path = md_path.with_suffix(".json")

    info = table_info()
    log(f"Table: {info['rows']:,} rows, indices: {[i['name'] for i in info['indices']]}")
    vec_idx = [i for i in info["indices"] if args.column in i["columns"]]
    if not vec_idx:
        sys.exit(f"No vector index on {args.column}: run `python app.py --create-index` first "
                 "(without it every query is a ~10s brute-force scan and nprobes/refine have no effect).")
    if vec_idx[0].get("unindexed"):
        log(f"WARNING: {vec_idx[0]['unindexed']:,} rows are not indexed yet (brute-forced on every query)")

    # 1-2. queries + ground truth (both cached across runs)
    queries = bench_prepare(args)
    truths = {None: bench_truth(args, queries, None), args.filter: bench_truth(args, queries, args.filter)}

    # 3. sweep
    sweep, baselines = [], []
    grid = [(n, r) for n in args.nprobes for r in args.refine]
    for i, (nprobes, refine) in enumerate(grid, 1):
        log(f"\n[{i}/{len(grid)}] nprobes={nprobes} refine_factor={refine}")
        sweep.append({
            "nprobes": nprobes, "refine": refine,
            "vector": measure("vector", vector_query(args, nprobes, refine), queries, truths[None], args),
            "vector_filter": measure("vector_filter", vector_query(args, nprobes, refine, args.filter),
                                     queries, truths[args.filter], args),
        })
        write_markdown(md_path, args, info, sweep, baselines, started)

    # 4. baselines
    if not args.skip_baselines:
        log("\n[baselines]")
        for name, fn in baseline_queries(args).items():
            baselines.append(measure(name, fn, queries, None, args))
            write_markdown(md_path, args, info, sweep, baselines, started)

    json_path.write_text(json.dumps({"args": vars(args), "table": info, "sweep": sweep,
                                     "baselines": baselines}, indent=2, default=str))
    log(f"\nReport -> {md_path}\nRaw    -> {json_path}")


if len(sys.argv) >= 2 and sys.argv[1] == '--benchmark':
    benchmark(sys.argv[2:])
elif len(sys.argv) == 2:
    if sys.argv[1] == '--create-index':
        list_indices = table.list_indices()

        if not any(filter(lambda i: i.name == "text_embedding_idx", list_indices)):
            table.create_index("text_embedding", config=IvfPq(distance_type="cosine", num_partitions=64, num_sub_vectors=16), name="text_embedding_idx")

        if not any(filter(lambda i: i.name == "text_idx", list_indices)):
            table.create_fts_index("text", language="English", replace=True, name="text_idx")

        table.wait_for_index(["text_embedding_idx", "text_idx"])
        print(table.list_indices());
    if sys.argv[1] == '--print':
        print(table.count_rows(), " rows")
        print(table.schema)
        print(table.search().select(["url", "text"]).limit(10).to_pandas())
elif len(sys.argv) == 3:
    if sys.argv[1] == '--text-search':
        results = table.search(query=MatchQuery(query=sys.argv[2], column="text"), query_type="fts", fts_columns="text").select(["id", "text", "url", "date", "language", "score"]).limit(5).to_pandas()
        print(results.drop(columns=["text_embedding"]))
    elif sys.argv[1] == '--vector-search':
        results = (table
            .search(embed_text(sys.argv[2]), vector_column_name="text_embedding")
            .distance_type("cosine")
            .select(["id", "text", "url", "date", "language", "score"])
            .limit(5)
            .to_pandas())
        print(results)
