from sqlalchemy.ext.asyncio.engine import create_async_engine

from sqlalchemy import text

from src.anubis.utils.configuration import GlobalConfiguration

import logging
logger = logging.getLogger(__name__)

from datetime import datetime

from dateutil import parser

# Convert string timestamps to datetime objects
def parse_datetime(dt_value):
    if isinstance(dt_value, datetime):
        return dt_value
    elif isinstance(dt_value, str):
        return parser.isoparse(dt_value)
    else:
        raise ValueError(f"Unexpected datetime type: {type(dt_value)}")
    

async def update_column_metadata(
    added_ids: list[str],
    creation_times_list: list[dict],
    configuration: GlobalConfiguration,
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
    
    async_engine = create_async_engine(configuration.vectorstore_postgres_uri)
    try:

        async with async_engine.connect() as conn:
            await conn.execute(text(SQL_UPDATE_METADATA), params)
            await conn.commit()
        return {"success": True}
    except Exception as e:
        logger.error("Error updating column metadata during 'update_column_metadata': {e}")
        return {"success": False}
    
from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_vector
from src.subgraphs.vector_store_graph.utils.helper_functions import update_column_metadata

async def batch_index_documents_vectorstore(
        id_list: list[str], 
        configuration: GlobalConfiguration, 
        vectorstore_documents_to_be_indexed: list[any], 
        BATCH_SIZE: int = 5000
    ):

        logger.info(f"BATCH INDEX DOCUMENTS VECTORSTORE BREAKPOINT")

        configuration = GlobalConfiguration()

        v_store = await make_pg_vector(configuration)

        total_documents_to_be_deleted = len(id_list)

        logger.info(f"id_list length: {total_documents_to_be_deleted}")

        # batch delete
        error_deleting_ids = []

        for i in range(0, total_documents_to_be_deleted, BATCH_SIZE):
            batch = id_list[i:i + BATCH_SIZE]
            batch_num  = (i // BATCH_SIZE) + 1
            total_batches = (total_documents_to_be_deleted + BATCH_SIZE - 1) // BATCH_SIZE
            try:
                await v_store.adelete(ids=batch) # there is no return value from a delete
                progress = min(i + BATCH_SIZE, total_documents_to_be_deleted)
                logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_deleted}")
            except Exception as e:
                print(f"Error in batch delete for batch number: {batch_num}: {str(e)}")
                # Handle the error with a retry
                logger.warning(f"error in delete: retrying...")
                try:
                    await v_store.adelete(ids=batch) # there is no return value from a delete
                    progress = min(i + BATCH_SIZE, total_documents_to_be_deleted)
                    logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_deleted}")
                except Exception as e:
                    logger.error(f"Continued Error in batch delete for batch number: {batch_num}: {str(e)}; returning error_ids_list.")
                    error_deleting_ids.extend(batch)
                    continue
        
        # Upload the new documents into the vector store
        logger.info(f"breakpoint before aadd documents")

        all_document_ids = []
        total_documents_to_be_indexed = len(vectorstore_documents_to_be_indexed)
        error_batch_documents = []

        # batch the document uploads
        for i in range(0, total_documents_to_be_indexed, BATCH_SIZE):
           
            batch = vectorstore_documents_to_be_indexed[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_documents_to_be_indexed + BATCH_SIZE - 1) // BATCH_SIZE
            try:
                batch_ids = await v_store.aadd_documents(batch)
                all_document_ids.extend(batch_ids)

                progress = min(i + BATCH_SIZE, total_documents_to_be_indexed)
                
                logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed}")
                
                # add columns for sorting

                logger.info(f"breakpoint after add to vectorstore")
                
                # Create creation_times_list:

                creation_times_list = [
                {
                    "created_at": doc.metadata['created_at'], 
                    "user_id":doc.metadata['user_id'], 
                    'assistant_id': doc.metadata['assistant_id'],
                    'filename': doc.metadata['filename']
                } for doc in batch]

                response = await update_column_metadata(
                    batch_ids, 
                    creation_times_list, 
                    configuration, 
                    table_name="langchain_pg_embedding"
                )

                logger.info(f"Update of column metadata: {response['success']}")
            except Exception as e:
                logger.info(f"Error in batch {batch_num}: {str(e)}. Attempting Retry.")
                # Handle the error with a retry
                try:
                    batch_ids = await v_store.aadd_documents(batch)
                    all_document_ids.extend(batch_ids)

                    progress = min(i + BATCH_SIZE, total_documents_to_be_indexed)

                    logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed}")

                    # add columns for sorting

                    logger.info(f"breakpoint after add to vectorstore")

                    # Create creation_times_list:

                    creation_times_list = [
                    {
                        "created_at": doc.metadata['created_at'], 
                        "user_id":doc.metadata['user_id'], 
                        'assistant_id': doc.metadata['assistant_id'],
                        'filename': doc.metadata['filename']
                    } for doc in batch]

                    response = await update_column_metadata(
                        batch_ids, 
                        creation_times_list, 
                        configuration, 
                        table_name="langchain_pg_embedding"
                    )

                    logger.info(f"Update of column metadata: {response['success']}")
                except Exception as e:
                    logger.info(f"Continued error in batch {batch_num}: {str(e)}. Returning batch documents in error")
                    error_batch_documents.extend(batch)
                    continue
        
        if len(error_batch_documents) == 0 and len(error_deleting_ids) == 0:
            return {"success": True}
        else:
            return {"success": False, "error_batch_documents": error_batch_documents, "error_deleting_ids": error_deleting_ids}
        