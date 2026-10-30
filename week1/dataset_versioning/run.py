import datetime
import sys
from time import perf_counter

import lance
import pandas as pd
import pyarrow as pa
import numpy as np

from sentence_transformers import SentenceTransformer
from lance.query import MatchQuery

K = 10
rng = np.random.default_rng(42)

queries = [
    "a heartwarming family movie with a happy ending",
    "terrible acting and a boring plot",
    "scary horror film with a lot of gore",
    "classic black and white film noir",
    "hilarious comedy that made me laugh out loud",
    "war movie about soldiers in Vietnam",
    "disappointing sequel that ruins the original",
    "beautiful animation and a great soundtrack",
    "low budget science fiction with cheesy special effects",
    "romantic drama about a love triangle",
    "documentary about politics and society",
    "martial arts action movie with great fight scenes",
    "musical with memorable songs and dancing",
    "slow paced art house film with no real story",
    "crime thriller with a twist ending",
    "superhero movie based on a comic book",
    "western with cowboys and gunfights",
    "adaptation of a Shakespeare play",
    "Christmas movie for kids",
    "biopic about a famous musician",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimensions = 384

schema = pa.schema([
    pa.field("text_movie", type=pa.string(), metadata=None),
    pa.field("score", type=pa.int16(), metadata=None),
    pa.field("embedding", type=pa.list_(pa.float64(), dimensions), metadata=None)
])

uri = "../load_dataset/text.lance"
ds = lance.dataset(uri)

def search(q, *, use_index, nprobes=16, refine_factor=4) -> set:
    nearest = {
        "column": "embedding",
        "q": q,
        "k": K,
        "metric": "cosine",  # must match the index metric
        "use_index": use_index,
    }
    if use_index:
        nearest.update(
            nprobes=nprobes,
            refine_factor=refine_factor,
        )

    return set(ds.scanner(
        columns=["_rowid"],
        nearest=nearest,
        disable_scoring_autoprojection=True,
    ).to_table()["_rowid"].to_pylist())
    
def benchmark(*, use_index, nprobes=16, refine_factor=4):
    options = dict(
        use_index=use_index,
        nprobes=nprobes,
        refine_factor=refine_factor,
    )

    # Warm-up
    for q in query_vectors[:5]:
        search(q, **options)

    latencies_ms, recalls = [], []

    for i in rng.permutation(len(query_vectors)):
        start = perf_counter()
        ids = search(query_vectors[i], **options)
        latencies_ms.append((perf_counter() - start) * 1000)
        recalls.append(len(set(ids) & truth[i]) / K)

    return {
        "recall@10": float(np.mean(recalls)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
    }

if len(sys.argv) >= 2:
    if sys.argv[1] == '--tag':
        if "sylvain" not in ds.tags.list():
            # Version 2 of my dataset
            tbl = pa.Table.from_pylist([{"text_movie": "Test Sylvain", "score": 1, "embedding": model.encode("Test Sylvain Insert")}], schema=schema)
            ds = lance.write_dataset(tbl, uri, mode="append")

            print(ds.count_rows(), "rows in the dataset")
            print(ds.version, " versions available")
            ds.tags.create("sylvain", 2)
        else:
            print("Checkout 'sylvain' tag")
            ds = ds.checkout_version(version="sylvain")
            print(ds.count_rows(), "rows in the dataset")
            print(ds.version, " versions available")
            print("Checkout by timestamp ", datetime.datetime(2026, 10, 26, 18, 00, 0))
            ds = lance.dataset(uri, asof=datetime.datetime(2026, 10, 26, 18, 00, 0))
            print(ds.count_rows(), "rows in the dataset")
            print(ds.version, " versions available")
    elif sys.argv[1] == '--branch':
        if "sylvain_branch" not in ds.branches.list():
            branch = ds.create_branch("sylvain_branch", "sylvain")
            tbl = pa.Table.from_pylist([{"text_movie": "Test Sylvain Branch", "score": 0, "embedding": model.encode("Test Sylvain Branch")}], schema=schema)
            lance.write_dataset(tbl, branch, mode="append")
            print("sylvain_branch created")
    elif sys.argv[1] == '--merge' and "sylvain_branch" in ds.branches.list():
        ds_sylvain = ds.checkout_version(("sylvain_branch", None))
        branch_data = ds_sylvain.to_table()
        ds = lance.write_dataset(branch_data, uri, mode="overwrite")
        print("'sylvain_branch' has been merged")
        print(ds.count_rows(), "rows in the dataset")
        print(ds.version, " versions available")
        ds.branches.delete("sylvain_branch")
        print("'sylvain_branch' is deleted")
    elif sys.argv[1] == '--index':
        ds = lance.dataset(uri)
        ds.create_index("embedding", index_type="IVF_PQ", num_partitions=64, num_sub_vectors=16, metric="cosine")
        print("Index created")
    elif sys.argv[1] == '--text-index':
        ds = lance.dataset(uri)
        ds.create_scalar_index("text_movie", index_type="INVERTED", replace=True, train=True)
        print("Full text index created")
    elif sys.argv[1] == '--search':
        ds = lance.dataset(uri)
        
        query_vectors = model.encode(queries)

        # Reference: exact (brute-force) results
        truth = [search(q, use_index=False) for q in query_vectors]
        
        print("Exact :", benchmark(use_index=False))

        for nprobes in [1, 4, 16, 64, 128]:
            for refine_factor in [1, 4, 16, 32]:
                result = benchmark(
                    use_index=True,
                    nprobes=nprobes,
                    refine_factor=refine_factor,
                )
                print(nprobes, refine_factor, result)
    elif sys.argv[1] == '--add':
        tbl = pa.Table.from_pylist([
            {"text_movie": "Test Add Text 1", "score": 0,
                "embedding": model.encode("Test Add Text 1")},
            {"text_movie": "Test Add Text 2", "score": 0,
                            "embedding": model.encode("Test Add Text 2")},
            {"text_movie": "Test Add Text 3", "score": 0,
                            "embedding": model.encode("Test Add Text 3")},
            {"text_movie": "Test Add Text 4", "score": 0,
                            "embedding": model.encode("Test Add Text 4")},
        ], schema=schema)
        ds.merge_insert("text_movie").when_not_matched_insert_all().execute(tbl)
   
        print(ds.count_rows(), "rows in the dataset")
        print(ds.version, " versions available")
    elif sys.argv[1] == '--text-search':
        start = perf_counter()
        result = ds.scanner(
            columns=["text_movie", "embedding", "_score"],
            full_text_query=MatchQuery(column="text_movie", query="Test Add Text 2"),
            limit=10).to_pandas()
        
        latencies_ms = (perf_counter() - start) * 1000
        
        print(result.head())
        print("Full Text Search Latency (without optimize_indices)= ", latencies_ms)
        
        if len(sys.argv) >= 3 and sys.argv[2] == '--optimize':
            start = perf_counter()
            ds.optimize.optimize_indices(retrain=False)
            result = ds.scanner(
                columns=["text_movie", "embedding", "_score"],
                full_text_query=MatchQuery(column="text_movie", query="Test Add Text 2"),
                limit=10).to_pandas()
            latencies_ms = (perf_counter() - start) * 1000
            print(result.head())
            print("Full Text Search Latency (with optimize_indices)= ", latencies_ms)