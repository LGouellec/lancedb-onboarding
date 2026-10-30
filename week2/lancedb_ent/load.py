import lance
import lancedb
from lancedb.pydantic import LanceModel, Vector

class CaptionModel(LanceModel):
    id: int
    image: bytes
    image_id: str
    filename: str
    captions: list[str]
    caption: str
    image_emb: Vector(dim=512)
    text_emb: Vector(dim=512)
    
ds = lance.dataset("hf://datasets/lance-format/coco-captions-2017-lance/data/val.lance")

print(ds.to_pandas().head().columns)

db = lancedb.connect("db://default", host_override="http://localhost:10024", api_key="b583083048a24627a7a3e6544bab26c337e0980f5d02736bdf2ab34c6f7754af")

table = db.create_table("val", data=ds, schema=CaptionModel, mode="overwrite")

print(table.to_pandas().head())