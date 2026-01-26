from fastapi import FastAPI, UploadFile, Form, HTTPException
from langgraph_sdk.server.fastapi import add_routes # Auto-agent routes
from chromadb import chromadb
from together import Together
from llama_api_client import LlamaAPIClient
from anubis import graph


app = FastAPI()

@app.post("/avatars/{avatar_id}/process-doc")
async def process_doc(avatar_id: str, file: UploadFile):
    content = await file.read()
    # analyze, format, vectorstore, update avatar, upload training data
    # content = await file.read().decode()
    
    # # Analyze (Together)
    # analysis = together_llm.invoke(f"Analyze: {content}")
    
    # # Update vectorstore
    # vectorstore.add_texts([analysis], {"avatar_id": avatar_id})
    
    # # Update Store (avatar features)
    # client.store.put(["avatars"], avatar_id, {
    #     "insights": analysis,
    #     "doc_count": client.store.get(["avatars"], avatar_id)["doc_count"] + 1
    # })
    analysis = {}
    
    return {"status": "processed", "analysis": analysis}

# Auto-add agent chat endpoints
add_routes(app, graph, path="/chat") # chat at /chat

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)