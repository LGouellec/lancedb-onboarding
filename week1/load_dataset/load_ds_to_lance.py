import pandas as pd
import pyarrow as pa
import lance
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dimensions = 384

def embed(sample, field='text'):
    return model.encode(inputs=sample[field])

schema = pa.schema([
    pa.field("text_movie", type=pa.string(), metadata=None),
    pa.field("score", type=pa.int16(), metadata=None),
    pa.field("embedding", type=pa.list_(pa.float64(), dimensions), metadata=None)
])

df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
df['embedding'] = df.apply(embed, axis=1)

df = df.rename(columns={"text": "text_movie", "label": "score"})

print(df.head())

table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
lance.write_dataset(table, "text.lance", schema=schema, mode="overwrite")

ds = lance.dataset("text.lance")

print(ds.count_rows())