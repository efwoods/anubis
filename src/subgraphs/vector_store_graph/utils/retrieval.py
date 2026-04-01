"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

import asyncio

import logging
logger = logging.getLogger(__name__)

## Encoder constructors
async def make_text_encoder(model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embeddings:
    """Connect to the configured text encoder."""
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info(f"Make text encoder  ENTRYPOINT")
    embeddings = await asyncio.to_thread(
        HuggingFaceEmbeddings,
        model_name=model,
        model_kwargs={"local_files_only": False, "trust_remote_code": False},
        # Optional: specify exact cache path
        cache_folder="src/models/"
    )

    return embeddings

