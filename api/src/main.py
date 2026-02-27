from typing import Optional

from fastapi import FastAPI

from langgraph_sdk import get_client

from supabase import Client

from contextlib import asynccontextmanager

from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)

import os
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info(f"lifespan initializing...")
        app.state.langgraph_sdk_client = get_client(url="https://anubis-prototype-8c36a9a2bd5a53c9a12d69bd22cc617b.us.langgraph.app", api_key=os.getenv("LANGSMITH_API_KEY"))
        yield

    except Exception as e:
        logger.error(f"Error: {e}")    
        raise e

app = FastAPI(
    title="Neural Nexus API", 
    description="API to interface with the Anubis graph",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/")
async def test_langgraph_sdk():
    langgraph_sdk_client = app.state.langgraph_sdk_client
    assistants = langgraph_sdk_client.assistants.search()
    langgraph_sdk_client
    logger.info(f"assitants: {assistants}")