import sys
from time import perf_counter

import lancedb
import lance
import numpy as np
import pandas as pd
import pyarrow as pa

from sentence_transformers import SentenceTransformer
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import IvfPq
from lancedb.query import MatchQuery

table_name = "demo"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimensions = 384
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


class Model(LanceModel):
    text_movie: str
    score: int
    embedding: Vector(dim=dimensions)
    
db = lancedb.connect("lancedb-demo")

def search(table: lancedb.table.LanceTable, q, *, use_index, nprobes=16, refine_factor=4) -> set:

    queryBuilder = (table
            .search(query=q, vector_column_name="embedding")
            .distance_type("cosine"))

    if use_index:
        queryBuilder = (queryBuilder
            .nprobes(nprobes)
            .refine_factor(refine_factor))
    else:
        queryBuilder = queryBuilder.bypass_vector_index()

    return set(queryBuilder.limit(K).to_pandas()["text_movie"])
    
def benchmark(table: lancedb.table.LanceTable, *, use_index, nprobes=16, refine_factor=4):
    options = dict(
        use_index=use_index,
        nprobes=nprobes,
        refine_factor=refine_factor,
    )

    # Warm-up
    for q in query_vectors[:5]:
        search(table, q, **options)

    latencies_ms, recalls = [], []

    for i in rng.permutation(len(query_vectors)):
        start = perf_counter()
        ids = search(table, query_vectors[i], **options)
        latencies_ms.append((perf_counter() - start) * 1000)
        recalls.append(len(ids & truth[i]) / K)

    return {
        "recall@10": float(np.mean(recalls)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
    }


if len(sys.argv) >= 2:
    if sys.argv[1] == '--init':
        ds = lance.dataset("text.lance")
        table = db.create_table(table_name, data=ds, schema=Model, mode="overwrite")
    elif sys.argv[1] == '--tag':
        table = db.open_table(table_name)
        if "sylvain" not in table.tags.list():
            tbl = pa.Table.from_pylist([{"text_movie": "Test Sylvain", "score": 1, "embedding": model.encode("Test Sylvain Insert")}])
            table.add(tbl)
            print(table.count_rows(), "rows in the dataset")
            print(table.version, " versions available")
            table.tags.create("sylvain", 2)
        else:
            print("Checkout 'sylvain' tag")
            table.checkout("sylvain")
            print(table.count_rows(), "rows in the dataset")
            print(table.version, " versions available")
    elif sys.argv[1] == '--branch':
        table = db.open_table(table_name)
        if "sylvain_branch" not in table.branches.list():
            branch = table.branches.create("sylvain_branch", from_version=2)
            tbl = pa.Table.from_pylist([{"text_movie": "Test Sylvain Branch", "score": 0, "embedding": model.encode("Test Sylvain Branch")}])
            branch.add(tbl)
            print("sylvain_branch created")
    elif sys.argv[1] == '--merge':
        main_table = db.open_table(table_name)
        if "sylvain_branch" in main_table.branches.list():
            table = db.open_table(table_name, branch="sylvain_branch")
            df = table.to_pandas()
            cleaned_df = df.drop_duplicates(subset=["text_movie"], keep="first")
            
            (main_table.merge_insert(on="text_movie")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(cleaned_df))
            
            print("'sylvain_branch' has been merged")
            print(main_table.count_rows(), "rows in the dataset")
            print(main_table.version, " versions available")
            main_table.branches.delete("sylvain_branch")
            print("'sylvain_branch' is deleted")
    elif sys.argv[1] == '--index':
        table = db.open_table(table_name)
        table.create_index("embedding", config=IvfPq(distance_type="cosine", num_partitions=64, num_sub_vectors=16))
        print("Index created")
    elif sys.argv[1] == '--text-index':
        table = db.open_table(table_name)
        table.create_fts_index("text_movie", language="English", replace=True)
        print("Full text index created")
    elif sys.argv[1] == '--search':
        
        table = db.open_table(table_name)
        query_vectors = model.encode(queries)

        # Reference: exact (brute-force) results
        truth = [search(table, q, use_index=False) for q in query_vectors]
        
        print("Exact :", benchmark(table, use_index=False))

        for nprobes in [1, 4, 16, 64, 128]:
            for refine_factor in [1, 4, 16, 32]:
                result = benchmark(
                    table,
                    use_index=True,
                    nprobes=nprobes,
                    refine_factor=refine_factor,
                )
                print(nprobes, refine_factor, result)
    elif sys.argv[1] == '--add':
        table = db.open_table(table_name)
        tbl = pa.Table.from_pylist([
            {"text_movie": "Test Add Text 1", "score": 0,
                "embedding": model.encode("Test Add Text 1")},
            {"text_movie": "Test Add Text 2", "score": 0,
                            "embedding": model.encode("Test Add Text 2")},
            {"text_movie": "Test Add Text 3", "score": 0,
                            "embedding": model.encode("Test Add Text 3")},
            {"text_movie": "Test Add Text 4", "score": 0,
                            "embedding": model.encode("Test Add Text 4")},
        ])
        table.merge_insert("text_movie").when_not_matched_insert_all().execute(tbl)
        print(table.count_rows(), "rows in the dataset")
        print(table.version, " versions available")
    elif sys.argv[1] == '--text-search':
        start = perf_counter()
        table = db.open_table(table_name)
        result = (
            table
            .search(query=MatchQuery(query="Test Add Text 2", column="text_movie"), query_type="fts", fts_columns="text_movie")
            .limit(K).to_pandas())
        
        latencies_ms = (perf_counter() - start) * 1000
        
        print(result.head())
        print("Full Text Search Latency (without optimize_indices)= ", latencies_ms)
        
        if len(sys.argv) >= 3 and sys.argv[2] == '--optimize':
            start = perf_counter()
            table.optimize()
            result = (
                        table
                        .search(query=MatchQuery(query="Test Add Text 2", column="text_movie"), query_type="fts", fts_columns="text_movie")
                        .limit(K).to_pandas())
            latencies_ms = (perf_counter() - start) * 1000
            print(result.head())
            print("Full Text Search Latency (with optimize_indices)= ", latencies_ms)