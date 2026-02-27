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
from src.anubis.utils.configuration import IndexConfiguration, GlobalConfiguration

import asyncio

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


@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", [])
    search_filter.append({"term": {"metadata.user_id": configuration.user_id}})
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", {})
    search_filter.update({"user_id": configuration.user_id})
    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=search_kwargs)


from src.anubis.utils.context import GlobalConfiguration

import logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def make_retriever(
    configuration: GlobalConfiguration,
) -> AsyncGenerator[VectorStoreRetriever, None]:
    """Create a retriever for the agent, based on the current configuration."""
    # configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = await make_text_encoder(configuration.embedding_model)

    logger.info(f" configuration.embedding_model: {configuration.embedding_model}")

    # embedding_model = HuggingFaceEmbeddings(configuration.embedding_model)
    # user_id = configuration.user_id
    # # assistant_id = configuration.assistant_id
    # if not user_id:
    #     raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(GlobalConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )

# from sqlalchemy.ext.asyncio import create_async_engine
# from langchain_postgres import PGVector

# async def make_pg_vector(
#     configuration: GlobalConfiguration):

#     """ EXAMPLE USAGE

#     vector_store = make_pg_vector(configuration)
#     async with vector_store as vector_store:
#         logger.info(f"breakpoint")
#         results = await vector_store.asimilarity_search_with_relevance_scores(
#         query="kitty", 
#         filter={"id": {"$in": [1,2,5,9]}},
#         score_threshold=0.3
#         )

#     retrieved_docs = [doc[0] for doc in results] # extract documents only; a tuple of relevance scores is returned

#     """

#     logger.info(f"make_pg_vector ENTRYPOINT")
#     embedding = await make_text_encoder(configuration.embedding_model)
#     logger.info(f"configuration.vectorstore_postgres_uri: {configuration.vectorstore_postgres_uri}")
#     async_engine = create_async_engine(configuration.vectorstore_postgres_uri)
        
#     vector_store = await asyncio.to_thread(
#         PGVector.from_existing_index,
#         embedding=embedding,
#         collection_name = "documents",
#         connection = async_engine,
#         async_mode=True,
#         create_extension=False
#     )

#     return vector_store


# === USE THIS METHOD WHEN CONNECTING TO A CUSTOM STORE === #

# from langgraph.store.postgres import AsyncPostgresStore

# import contextlib
# from langgraph.store.base import IndexConfig


# @contextlib.asynccontextmanager
# async def make_pg_store():

#     logger.info(f"make_pg_store ENTRYPOINT")

#     configuration = GlobalConfiguration()
#     embeddings = await make_text_encoder(configuration.embedding_model)

#     async with AsyncPostgresStore.from_conn_string(
#         conn_string=configuration.async_postgres_store_uri,
#         index = IndexConfig(dims=384, embed=embeddings, fields=["page_content"])
#         ) as store:
#         await store.setup()
#         yield store

