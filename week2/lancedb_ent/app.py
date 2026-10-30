from datetime import datetime, timedelta
import re
import sys
from pathlib import Path

import lancedb
from lancedb.table import IvfPq
from lancedb.query import MatchQuery

db = lancedb.connect("db://default", host_override="http://localhost:10024", api_key="b583083048a24627a7a3e6544bab26c337e0980f5d02736bdf2ab34c6f7754af")

print(db.list_tables())

val = db.open_table("val")

# image_emb / text_emb were generated with OpenCLIP ViT-B/32 trained on LAION-2B (512-dim, L2-normalized)
clip_model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def embed_text(text):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(clip_model_name).eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    with torch.no_grad():
        features = model.get_text_features(**tokenizer([text], return_tensors="pt", padding=True))
    features = getattr(features, "pooler_output", features)
    return torch.nn.functional.normalize(features, dim=-1)[0].numpy()


def save_images(results, query):
    # One folder per request: results/<timestamp>_<query>/
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query).strip("_")[:50]
    out_dir = Path("results") / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in results.itertuples():
        (out_dir / f"{row.id}.jpg").write_bytes(row.image)
    print(f"Saved {len(results)} images to {out_dir}")


if len(sys.argv) == 2:
    if sys.argv[1] == '--create-index':
        val.create_index("image_emb", config=IvfPq(distance_type="cosine", num_partitions=64, num_sub_vectors=16), name="image_emb_idx")
        val.create_fts_index("caption", language="English", replace=True, name="caption_idx")
        val.wait_for_index(["image_emb_idx", "caption_idx"])
        print(val.list_indices());
    if sys.argv[1] == '--print':
        print(val.schema)
        print(val.search().limit(10).to_pandas())
elif len(sys.argv) == 3:
    if sys.argv[1] == '--text-search':
        results = val.search(query=MatchQuery(query=sys.argv[2], column="caption"), query_type="fts", fts_columns="caption").select(["id", "caption", "captions", "image"]).limit(5).to_pandas()
        print(results.drop(columns=["image"]))
        save_images(results, sys.argv[2])
    elif sys.argv[1] == '--vector-search':
        # Text -> image: embed the query with the same CLIP model and search the image embeddings
        results = (val
            .search(embed_text(sys.argv[2]), vector_column_name="image_emb")
            .distance_type("cosine")
            .select(["id", "caption","captions", "image"])
            .limit(5)
            .to_pandas())
        print(results.drop(columns=["image"]))
        save_images(results, sys.argv[2])
