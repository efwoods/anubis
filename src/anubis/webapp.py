import os
from fastapi import FastAPI, UploadFile
from src.anubis.graph import graph

from fastapi import FastAPI

app = FastAPI(title="Debug API")

@app.get("/hello")
def test_hello_world():
    return {"Hello":"World"}

@app.get("/world")
def test_world():
    return {"This": "Works"}


# @app.post("/invoke")
# async def invoke_graph(body: {}):
#     config = {"configurable": {"thread_id": "test"}}
#     result = graph.invoke(body.get("input", {"messages":[]}), config)
#     return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
