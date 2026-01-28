import os
from fastapi import FastAPI, UploadFile
from src.agent.graph import graph



from fastapi import FastAPI

app = FastAPI(title="Debug API")

@app.get("/hello")
def test_hello_world():
    return {"Hello":"World"}

@app.get("/world")
def test_world():
    return {"This": "Works"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
