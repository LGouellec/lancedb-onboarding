import random
from typing import Optional

import lancedb
from lancedb import Table, col, udf
import pyarrow as pa

TABLE_NAME = "functions_demo"
NUM_ROWS = 5000

WORDS = ["lance", "vector", "search", "index", "query", "table", "column",
            "arrow", "data", "fast", "storage", "embedding", "score", "row"]

db = lancedb.connect(
    "db://function",
    host_override="http://localhost:10024",
    api_key="b583083048a24627a7a3e6544bab26c337e0980f5d02736bdf2ab34c6f7754af")


def create_table(name=TABLE_NAME, num_rows=NUM_ROWS, seed=42):
    """Create (or recreate) a table with columns id, text, score filled with `num_rows` rows."""
    if name in db.list_tables().tables:
        db.drop_table(name)
        print(f"Dropped existing table '{name}'")

    rng = random.Random(seed)
    data = pa.table({
        "id": pa.array(range(num_rows), pa.int64()),
        "text": pa.array(
            [" ".join(rng.choices(WORDS, k=rng.randint(3, 15))) for _ in range(num_rows)],
            pa.string()),
        "score": pa.array([rng.random() for _ in range(num_rows)], pa.float64()),
    })
    table = db.create_table(name, data)
    print(f"Created table '{name}' with {table.count_rows()} rows")
    return table


@udf
def text_length(text: Optional[str]) -> int:
    """Number of characters in `text`, computed server-side for each row."""
    return  None if text is 0 else len(text)


def add_text_length_column(table: Table):
    """Register the `text_length` UDF, add it as a Function column, then backfill it."""
    fn = db.create_function(text_length)
    print(f"Registered function '{fn.name}' version {fn.version}")

    if "text_length" not in table.schema.names:
        table.add_columns({"text_length": fn(text=col("text"))})

    result = table.refresh_column_async("text_length").wait()
    print(f"Backfilled 'text_length': {result}")
    return table


if __name__ == "__main__":
    table = create_table()
    print(table.schema)
    add_text_length_column(table)
    print(table.schema)
    print(table.search().select(["id", "text", "score", "text_length"]).limit(10).to_pandas())
