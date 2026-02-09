from langgraph.store.postgres import PostgresStore

connection_string = ""
POSTGRES_DB_URI="postgresql://postgres:postgres@127.0.0.1:54322"

with PostgresStore.from_conn_string(POSTGRES_DB_URI) as postgres_store:
    postgres_store.setup()
