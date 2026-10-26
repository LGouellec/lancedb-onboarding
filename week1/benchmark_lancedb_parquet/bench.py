import lance
import pyarrow as pa
import pyarrow.dataset  # noqa: F401 (needed for pa.dataset)
import random
import pandas as pd
import string
import sys
import time
import numpy as np
import lance

batch_size = 1_000_000


def generate_ds():
    for i in range(100):
        print(f"Creating batch {i}")
        string_arr = pa.array([''.join(random.choices(
            string.ascii_letters, k=1_000)) for _ in range(batch_size)])
        tbl = pa.Table.from_arrays([string_arr], names=["value"])
        print(f"Writing batch {i} to lance")
        if i == 0:
            params = {"mode": "create"}
        else:
            params = {"mode": "append"}
        lance.write_dataset(tbl, "take.lance", **params)

    ds = lance.dataset("take.lance")
    pa.dataset.write_dataset(ds.scanner().to_reader(),
                             "take.parquet",
                             format="parquet")


def run_benchmark_lance():
    tot_time = 0
    tot_keys = 0
    nruns = 1000
    ds = lance.dataset("take.lance")

    for _ in range(nruns):
        nrows = np.random.randint(20, 50, 1)
        row_ids = np.random.randint(0, 100_000_000, nrows)
        start = time.time()
        tbl = ds.take(row_ids)
        end = time.time()
        tot_time += end - start
        tot_keys += len(row_ids)

    print(f"Lance: mean time per key is {tot_time / tot_keys}")


def run_benchmark_parquet():
    tot_time = 0
    tot_keys = 0
    nruns = 1000
    ds = pd.read_parquet("take.parquet")

    for _ in range(nruns):
        nrows = np.random.randint(20, 50, 1)
        row_ids = np.random.randint(0, 100_000_000, nrows)
        start = time.time()
        tbl = ds.take(row_ids)
        end = time.time()
        tot_time += end - start
        tot_keys += len(row_ids)

    print(f"Parquet: mean time per key is {tot_time / tot_keys}")


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        generate_ds()
    else:
        run_benchmark_lance()
        run_benchmark_parquet()
    # benchmark continues here
