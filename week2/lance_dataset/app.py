import os

import lance

source_uri = "hf://datasets/lance-format/coco-captions-2017-lance/data/val.lance"

container = os.environ["AZURE_STORAGE_CONTAINER"]
target_uri = f"az://{container}/default/val.lance"

storage_options = {
    "account_name": os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
    "account_key": os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
}

ds = lance.dataset(source_uri)
print(f"Source: {ds.count_rows()} rows")
print(ds.schema)

# Stream batches from Hugging Face to Azure instead of loading everything in memory
lance.write_dataset(
    ds.scanner(batch_size=1024).to_reader(),
    target_uri,
    schema=ds.schema,
    mode="overwrite",
    storage_options=storage_options,
)

azure_ds = lance.dataset(target_uri, storage_options=storage_options)
print(f"Azure: {azure_ds.count_rows()} rows at {target_uri}")
print(azure_ds.to_table(columns=["image_id", "caption"], limit=5).to_pandas())
