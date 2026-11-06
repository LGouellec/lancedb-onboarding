import lancedb

db = lancedb.connect(
    "db://benchv2",
    host_override="http://localhost:10024",
    api_key="b583083048a24627a7a3e6544bab26c337e0980f5d02736bdf2ab34c6f7754af")

print(db.list_tables().tables)

table = db.open_table("bench_20M")

print(table.count_rows())
print(table.schema)

r = table.search().select(["id", "text", "category", "vector"]).limit(10).to_pandas()
res = (
    table.search(r['vector'][0], vector_column_name='vector')
    .bypass_vector_index()
    .select(['id','vector', "_distance"])
    .limit(10).to_pandas())

print('Query : ' , r['vector'][0])
print('Result : ', res)