from sqlalchemy.ext.asyncio.engine import create_async_engine

from sqlalchemy import text

from src.anubis.utils.context import GlobalContext

import logging
logger = logging.getLogger(__name__)

from datetime import datetime

from dateutil import parser
from uuid import NAMESPACE_URL, uuid5

from langgraph.config import get_stream_writer


def _emit_index_progress(batch_num: int, total_batches: int, indexed: int, total: int) -> None:
    """Emit an indexing ``media_progress`` event for the background-job SSE stream.

    Safe no-op outside a streaming context (e.g. plain ``ainvoke``).
    """
    try:
        writer = get_stream_writer()
        writer(
            {
                "type": "media_progress",
                "stage": "indexing",
                "current": batch_num,
                "total": total_batches,
                "documents_indexed": indexed,
                "documents_total": total,
            }
        )
    except Exception:  # pragma: no cover - progress must never break indexing
        pass

# Convert string timestamps to datetime objects
def parse_datetime(dt_value):
    if isinstance(dt_value, datetime):
        return dt_value
    elif isinstance(dt_value, str):
        return parser.isoparse(dt_value)
    else:
        raise ValueError(f"Unexpected datetime type: {type(dt_value)}")
    
# Delete all chunks of a document by filename
import logging
logger = logging.getLogger(__name__)

async def delete_docs_filename(runtime, user_id, assistant_id, filenames):
    idx = 0
    for filename in filenames:
        logger.info(f"Total remaining files to be deleted: {len(filenames) - idx}")
        
        namespace = (user_id, assistant_id, "document", filename)
        
        search_results = await runtime.store.asearch(namespace)
        del_list = [(res.namespace, res.key) for res in search_results]
        total_items_to_be_deleted = len(search_results)
        
        logger.info(f"deleting {total_items_to_be_deleted} items associated with {filename}")
        [await runtime.store.adelete(namespace=item[0], key=item[1]) for item in (del_list)]
        
        logger.info(f"deleted {len(search_results)} items associated with {filename}")
        idx+=1


async def update_column_metadata(
    added_ids: list[str],
    creation_times_list: list[dict],
    context: GlobalContext,
    table_name: str = "langchain_pg_embedding"
):
    
    logger.info("update_column_metadata ENTRYPOINT")
    
    if not added_ids or not creation_times_list:
        return
    

    assert len(added_ids) == len(creation_times_list)
    
    SQL_UPDATE_METADATA = f"""
    UPDATE {table_name} d
    SET 
        user_id = v.user_id,
        assistant_id = v.assistant_id,
        created_at = v.created_at::timestamptz,
        filename = v.filename
    FROM (
        SELECT 
            unnest(CAST(:ids AS varchar[])) as id,
            unnest(CAST(:user_ids AS text[])) as user_id,
            unnest(CAST(:assistant_ids AS text[])) as assistant_id,
            unnest(CAST(:created_ats AS timestamptz[])) as created_at,
            unnest(CAST(:filenames AS text[])) as filename
    ) AS v
    WHERE d.id = v.id;
    """
    
    params = {
        'ids': added_ids,
        'user_ids': [m['user_id'] for m in creation_times_list],
        'assistant_ids': [m['assistant_id'] for m in creation_times_list],
        'created_ats': [parse_datetime(m['created_at']) for m in creation_times_list],
        'filenames': [m['filename'] for m in creation_times_list]
    }
    
    async_engine = create_async_engine(context.vectorstore_postgres_uri)
    try:

        async with async_engine.connect() as conn:
            await conn.execute(text(SQL_UPDATE_METADATA), params)
            await conn.commit()
        return {"success": True}
    except Exception as e:
        logger.error("Error updating column metadata during 'update_column_metadata': {e}")
        return {"success": False}
    
# from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_vector
from src.subgraphs.vector_store_graph.utils.helper_functions import update_column_metadata
from langchain_core.stores import BaseStore
import uuid
from langgraph.store.base import PutOp

from langgraph.runtime import Runtime
from uuid import uuid4

async def batch_index_documents_vectorstore(
        store: BaseStore,
        user_id: str, 
        assistant_id: str,
        vectorstore_documents_to_be_indexed: list[any], 
        BATCH_SIZE: int = 1000
    ):

        logger.info(f"BATCH INDEX DOCUMENTS VECTORSTORE BREAKPOINT")

         # Delete documents with the same filename in the metadata
        filenames = [
            doc.metadata.get("namespace_filename") 
            for doc in 
            vectorstore_documents_to_be_indexed
            if doc.metadata.get("namespace_filename") is not None
        ]

        # Ensure each document has a unique key (str document_id):
        for doc in vectorstore_documents_to_be_indexed:
            existing = doc.metadata.get("document_id")
            if not isinstance(existing, str) or not existing:
                doc.metadata["document_id"] = str(uuid.uuid4())

        # acquire keys
        insert_document_keys = [
            doc.metadata.get("document_id", "")
            for doc in 
            vectorstore_documents_to_be_indexed
        ]

        try:
            assert len(insert_document_keys) == len(vectorstore_documents_to_be_indexed)  == len(filenames)
        except Exception as e:
            logger.warning(f"Assertion error: number of document keys, documents, and document filenames are not equal during document batch indexing:\n {e}")
            

        # Replace-by-filename at commit time. Each document is keyed by a random
        # document_id (above), so a re-run would otherwise append duplicate rows.
        # Remove any existing rows for the incoming namespace_filename(s) here —
        # immediately before inserting the new set — rather than deleting up
        # front in the request handler. This way a crash during media processing
        # never destroys the prior version: the old documents are only removed at
        # the moment the new ones are written. Idempotent on the deterministic
        # namespace_filename (uuid5), so resumed/re-run items overwrite cleanly.
        incoming_filenames = {f for f in filenames if f}
        if incoming_filenames:
            try:
                existing_items = await store.asearch(
                    (user_id, assistant_id), limit=1_000_000
                )
            except Exception as e:
                logger.warning(
                    "replace-by-filename search failed (continuing without delete): %s",
                    e,
                )
                existing_items = []

            to_delete: list[tuple] = []
            for item in existing_items or []:
                value = getattr(item, "value", None)
                if not isinstance(value, dict):
                    continue
                document = value.get("document")
                kwargs_blob = document.get("kwargs") if isinstance(document, dict) else None
                metadata = kwargs_blob.get("metadata") if isinstance(kwargs_blob, dict) else None
                stored_filename = (
                    metadata.get("namespace_filename") if isinstance(metadata, dict) else None
                )
                is_reference_audio = metadata.get("reference_audio", False)
                is_reference_image = metadata.get("reference_image", False)
                if isinstance(stored_filename, str) and stored_filename in incoming_filenames and not (is_reference_audio or is_reference_image):
                    to_delete.append((item.namespace, item.key))

            for ns, key in to_delete:
                try:
                    await store.adelete(namespace=ns, key=key)
                except Exception as e:
                    logger.warning(
                        "replace-by-filename delete failed for %s/%s: %s", ns, key, e
                    )
            if to_delete:
                logger.info(
                    "Replaced %d existing row(s) for %d filename(s) before indexing",
                    len(to_delete),
                    len(incoming_filenames),
                )

        # Upload the new documents into the vector store
        logger.info(f"breakpoint before aadd documents")

        # create upload namespaces
        batch_put_ops = [PutOp(namespace=(user_id, assistant_id, doc.metadata.get("namespace","document") , doc.metadata.get("namespace_filename", f"{user_id}'_'{assistant_id}'_document_unknown_filename'")), key=key, value={"document":doc.to_json()}) for key, doc in zip(insert_document_keys, vectorstore_documents_to_be_indexed)]

        num_successful_batch_uploads = 0
        total_documents_to_be_indexed = len(batch_put_ops)
        error_batch_documents = []

        # batch the document uploads
        for i in range(0, total_documents_to_be_indexed, BATCH_SIZE):
           
            batch = batch_put_ops[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_documents_to_be_indexed + BATCH_SIZE - 1) // BATCH_SIZE
            try:
                await store.abatch(batch)

                progress = min(i + BATCH_SIZE, total_documents_to_be_indexed)

                logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed}")
                _emit_index_progress(batch_num, total_batches, progress, total_documents_to_be_indexed)

                num_successful_batch_uploads += len(batch)

            except Exception as e:
                logger.info(f"Error in batch {batch_num}: {str(e)}. Attempting Retry.")
                # Handle the error with a retry
                try:
                    await store.abatch(batch)

                    progress = min(i + BATCH_SIZE, total_documents_to_be_indexed)

                    logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed}")

                except Exception as e:
                    logger.info(f"Continued error in batch {batch_num}: {str(e)}. Returning batch documents in error")
                    error_batch_documents.extend(batch)
                    continue
        
        if len(error_batch_documents) == 0:
            return {"success": True, "documents_uploaded": num_successful_batch_uploads}
        else:
            return {"success": False, "error_batch_documents": error_batch_documents}
        