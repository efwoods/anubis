from sqlalchemy.ext.asyncio.engine import create_async_engine

from sqlalchemy import text

from src.anubis.utils.configuration import GlobalConfiguration

import logging
logger = logging.getLogger(__name__)

async def update_column_metadata(
    added_ids: list[str],
    creation_times_list: list[dict],
    configuration: GlobalConfiguration,
    table_name: str = "langchain_pg_embedding"
):
    if not added_ids or not creation_times_list:
        return
    
    assert len(added_ids) == len(creation_times_list)
    
    SQL_UPDATE_METADATA = f"""
    UPDATE {table_name} d
    SET 
        user_id = v.user_id,
        assistant_id = v.assistant_id,
        created_at = v.created_at::timestamptz
    FROM (
        SELECT 
            unnest(:ids) as id,
            unnest(:user_ids) as user_id,
            unnest(:assistant_ids) as assistant_id,
            unnest(:created_ats) as created_at
    ) AS v
    WHERE d.id = v.id;
    """
    
    params = {
        'ids': added_ids,
        'user_ids': [m['user_id'] for m in creation_times_list],
        'assistant_ids': [m['assistant_id'] for m in creation_times_list],
        'created_ats': [m['created_at'] for m in creation_times_list]
    }
    
    async_engine = create_async_engine(configuration.postgres_uri)
    try:

        async with async_engine.connect() as conn:
            await conn.execute(text(SQL_UPDATE_METADATA), params)
            await conn.commit()
        return {"success": True}
    except Exception as e:
        logger.error("Error updating column metadata during 'update_column_metadata': {e}")
        return {"success": False}