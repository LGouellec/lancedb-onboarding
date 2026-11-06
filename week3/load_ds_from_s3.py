"""Copy a Lance dataset stored on S3 into a LanceDB (Enterprise) table, in resumable chunks.

Usage:
    pip install pylance tqdm
    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1   # or AWS_PROFILE
    python load_ds_from_s3.py s3://my-bucket/path/dataset.lance my_table [--chunk-mb 512] [--overwrite]

The source is streamed in order and committed to the target every ~--chunk-mb of data,
so an interrupted copy can be resumed: re-running the same command skips the rows that
are already in the target table (it assumes the target was only filled by this script).
"""

import argparse
import os
import time
import lance
import lancedb
import pyarrow as pa
from tqdm import tqdm

db = lancedb.connect("db://bench", host_override="http://localhost:10024", api_key="b583083048a24627a7a3e6544bab26c337e0980f5d02736bdf2ab34c6f7754af")


def s3_storage_options():
    # Lance picks up the standard AWS env vars / profile on its own; only pass what's explicitly set.
    opts = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
        "aws_region": os.getenv("AWS_REGION"),
        "aws_endpoint": os.getenv("AWS_ENDPOINT_URL"),  # e.g. MinIO
    }
    return {k: v for k, v in opts.items() if v}


RESERVED_METADATA_PREFIXES = (b"virtual_column",)


def clean_schema(schema):
    """Drop field metadata the server reserves for its own computed columns (e.g. Geneva's
    `virtual_column.udf_inputs`). The column data is kept; it just becomes a regular column."""
    fields = []
    for field in schema:
        meta = {k: v for k, v in (field.metadata or {}).items() if not k.startswith(RESERVED_METADATA_PREFIXES)}
        fields.append(field.with_metadata(meta) if meta else field.remove_metadata())
    return pa.schema(fields, metadata=schema.metadata)


def chunks(reader, schema, max_bytes):
    """Group record batches from `reader` into pyarrow Tables of roughly `max_bytes` each."""
    batches, size = [], 0
    for batch in reader:
        batch = pa.RecordBatch.from_arrays(batch.columns, schema=schema)  # same data, cleaned metadata
        batches.append(batch)
        size += batch.nbytes
        if size >= max_bytes:
            yield pa.Table.from_batches(batches)
            batches, size = [], 0
    if batches:
        yield pa.Table.from_batches(batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_uri", help="S3 URI of the Lance dataset, e.g. s3://bucket/path/data.lance")
    parser.add_argument("table", help="Target LanceDB table name")
    parser.add_argument("--overwrite", action="store_true", help="Drop the target table and copy from scratch")
    parser.add_argument("--chunk-mb", type=int, default=512, help="Approximate size of each commit (MB)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Rows per batch read from S3")
    parser.add_argument("--version", type=int, default=None, help="Source dataset version to copy (default: latest)")
    args = parser.parse_args()

    # 1. Open the source dataset on S3 (pin the version so a resume reads the exact same data)
    src = lance.dataset(args.source_uri, version=args.version, storage_options=s3_storage_options())
    total_rows = src.count_rows()
    print(f"Source: {args.source_uri} (version {src.version}), {total_rows:,} rows, {len(src.get_fragments())} fragments")

    # 2. Create the target table, or resume into the existing one
    schema = clean_schema(src.lance_schema.to_pyarrow())
    if args.table in db.table_names() and args.overwrite:
        db.drop_table(args.table)
    if args.table in db.table_names():
        tbl = db.open_table(args.table)
        if not tbl.schema.equals(schema, check_metadata=False):
            raise SystemExit(f"Table '{args.table}' exists with a different schema; use --overwrite.\n{tbl.schema}")
    else:
        tbl = db.create_table(args.table, schema=schema)

    offset = tbl.count_rows()
    if offset > total_rows:
        raise SystemExit(f"Target already has {offset:,} rows (> source {total_rows:,}); use --overwrite.")
    if offset == total_rows:
        print(f"Nothing to do: '{args.table}' already has all {total_rows:,} rows.")
        return
    if offset:
        print(f"Resuming: '{args.table}' already has {offset:,} rows, copying the remaining {total_rows - offset:,}")
        print(f"(If this was not created by an earlier run of this script for version {src.version}, use --overwrite.)")

    # 3. Stream the remaining rows in order and commit one chunk at a time
    reader = src.scanner(offset=offset, limit=total_rows - offset, batch_size=args.batch_size, scan_in_order=True).to_reader()
    start, written_bytes, n_commits = time.time(), 0, 0
    with tqdm(total=total_rows, initial=offset, unit="rows", unit_scale=True, desc=f"S3 -> {args.table}") as pbar:
        for chunk in chunks(reader, schema, args.chunk_mb * 1024 * 1024):
            committed = pbar.n

            def on_progress(p):
                # live progress inside the current chunk
                pbar.n = committed + p["output_rows"]
                pbar.refresh()

            result = tbl.add(chunk, progress=on_progress)
            n_commits += 1
            written_bytes += chunk.nbytes
            pbar.n = committed + chunk.num_rows
            elapsed = time.time() - start
            pbar.set_postfix(commits=n_commits, version=result.version, GB=f"{written_bytes / 1e9:.2f}", MBps=f"{written_bytes / 1e6 / elapsed:.1f}")

    # 4. Verify
    final = tbl.count_rows()
    print(f"Done in {time.time() - start:.1f}s: {n_commits} commits, table has {final:,}/{total_rows:,} rows")
    if final != total_rows:
        raise SystemExit("Row count mismatch between source and target!")


if __name__ == "__main__":
    main()
