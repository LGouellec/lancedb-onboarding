"""Generate a mock dataset and write it to S3 in the Lance format.

Usage:
    pip install pylance numpy tqdm
    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1   # or AWS_PROFILE
    python load_ds_to_s3.py s3://my-bucket/path/dataset.lance [--rows 100000] [--dim 128] [--overwrite]

The data is generated and uploaded batch by batch, so memory stays flat whatever --rows is.
The result can be read back with load_ds_from_s3.py.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import lance
import numpy as np
import pyarrow as pa
from tqdm import tqdm

CATEGORIES = ["books", "electronics", "garden", "music", "sports", "toys"]
WORDS = ["lance", "vector", "search", "table", "index", "query", "fast", "columnar", "cloud", "storage", "random", "mock"]


def s3_storage_options():
    # Lance picks up the standard AWS env vars / profile on its own; only pass what's explicitly set.
    opts = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
        "aws_region": os.getenv("AWS_REGION"),
        "aws_endpoint": os.getenv("AWS_ENDPOINT_URL"),  # e.g. MinIO
    }
    opts = {k: v for k, v in opts.items() if v}
    if opts.get("aws_endpoint", "").startswith("http://"):
        opts["allow_http"] = "true"
    return opts


def mock_schema(dim):
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("text", pa.string()),
            pa.field("category", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("in_stock", pa.bool_()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def mock_batches(schema, rows, dim, batch_size, seed, start_id=0, pbar=None):
    """Yield record batches of random rows, with ids start_id..start_id+rows-1."""
    rng = np.random.default_rng(seed)
    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for start in range(start_id, start_id + rows, batch_size):
        n = min(batch_size, start_id + rows - start)
        words = rng.choice(WORDS, size=(n, 8))
        seconds = rng.integers(0, 365 * 24 * 3600, size=n)
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(np.arange(start, start + n, dtype=np.int64)),
                pa.array([" ".join(w) for w in words]),
                pa.array(rng.choice(CATEGORIES, size=n)),
                pa.array(np.round(rng.uniform(1, 500, size=n), 2)),
                pa.array(rng.random(n) < 0.8),
                pa.array([epoch + timedelta(seconds=int(s)) for s in seconds], type=schema.field("created_at").type),
                pa.FixedSizeListArray.from_arrays(pa.array(rng.random(n * dim, dtype=np.float32)), dim),
            ],
            schema=schema,
        )
        if pbar is not None:
            pbar.update(n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target_uri", help="S3 URI of the Lance dataset to create, e.g. s3://bucket/path/data.lance")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows to generate")
    parser.add_argument("--dim", type=int, default=128, help="Dimension of the vector column")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Rows generated per batch")
    parser.add_argument("--max-rows-per-file", type=int, default=1_000_000, help="Max rows per Lance data file (fragment)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", help="Replace the dataset if it already exists")
    group.add_argument("--append", action="store_true", help="Append to the dataset if it already exists")
    args = parser.parse_args()

    storage_options = s3_storage_options()
    schema = mock_schema(args.dim)
    mode = "overwrite" if args.overwrite else "append" if args.append else "create"

    # When appending, continue the ids after the existing rows (and vary the seed so rows differ).
    start_id = 0
    if args.append:
        try:
            start_id = lance.dataset(args.target_uri, storage_options=storage_options).count_rows()
        except ValueError:
            mode = "create"  # nothing there yet

    with tqdm(total=args.rows, unit="rows", unit_scale=True, desc=f"mock -> {args.target_uri}") as pbar:
        batches = mock_batches(schema, args.rows, args.dim, args.batch_size, args.seed + start_id, start_id, pbar)
        try:
            ds = lance.write_dataset(
                pa.RecordBatchReader.from_batches(schema, batches),
                args.target_uri,
                mode=mode,
                max_rows_per_file=args.max_rows_per_file,
                storage_options=storage_options,
            )
        except OSError as e:
            if mode == "create" and "already exists" in str(e).lower():
                raise SystemExit(f"Dataset already exists at {args.target_uri}; use --overwrite or --append.")
            raise

    print(f"Done: {args.target_uri} (version {ds.version}), {ds.count_rows():,} rows, {len(ds.get_fragments())} fragments")
    print(ds.schema)


if __name__ == "__main__":
    main()
