# Feature engineering + backfill with Geneva on the cluster's KubeRay.
#
# Prereqs: terraform apply with enable_geneva = true, kuberay-operator and the
# geneva chart installed, `az login` as a principal listed in
# geneva_client_principal_ids, and kubectl pointed at the cluster.
#
#   python3.12 -m venv .venv && . .venv/bin/activate
#   pip install "geneva==0.16.0"
#   export AZURE_STORAGE_ACCOUNT_NAME=palantirnews
#   python build/geneva_backfill.py

import sys

import pyarrow as pa
import ray
import geneva
from geneva.cluster.builder import KubeRayClusterBuilder
from geneva.runners.ray.raycluster import ExitMode

DB_URI = "az://lancedb/default"
TABLE = "val"
FEATURE_COLUMN = "caption_length"


@geneva.udf(data_type=pa.int32())
def caption_length(caption: str | None) -> int:
    return len(caption) if caption is not None else 0


db = geneva.connect(DB_URI)

print(db.table_names())

table = db.open_table(TABLE)
if FEATURE_COLUMN not in table.schema.names:
    table.add_columns({FEATURE_COLUMN: caption_length})

# Head and workers land on the geneva node pool through Geneva's default node
# selectors and run as geneva-service-account (workload identity).
# Geneva derives the default image from the driver's CPU, so an Apple Silicon
# laptop would pick an arm64 image; the geneva node pool is amd64.
RAY_IMAGE = f"rayproject/ray:{ray.__version__}-py{sys.version_info.major}{sys.version_info.minor}"

cluster = (
    KubeRayClusterBuilder.create("feature-backfill")
    .namespace("lancedb")
    .head_group(image=RAY_IMAGE)
    .add_worker_group(
        KubeRayClusterBuilder.cpu_worker()
        .image(RAY_IMAGE)
        .cpus(4)
        .memory("16Gi")
        .max_replicas(4)
        .build()
    )
    .build()
)
db.define_cluster("feature-backfill", cluster)

with db.context(cluster="feature-backfill", on_exit=ExitMode.RETAIN_ON_FAILURE):
    result = table.backfill(FEATURE_COLUMN, where=f"{FEATURE_COLUMN} IS NULL")
    print(result)
