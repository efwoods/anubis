

""" CSV """

from langchain_core.messages.utils import count_tokens_approximately

from pathlib import Path
import sys
import os
from langchain_community.document_loaders import CSVLoader

from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langsmith import Client
import pandas as pd

from src.anubis.utils.model import init_model
from src.anubis.utils.context import GlobalContext
from rich.markdown import Markdown
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import BaseModel
from typing import List, Literal
from langchain_core.messages.utils import count_tokens_approximately
from uuid import NAMESPACE_URL, uuid4, uuid5
from pathlib import Path

async def csv_to_doc(file_path):
    """ Parse a CSV to a list of Document Objects. Expects a list of text from the original speaker only. """

    df = pd.read_csv(file_path)    
    
    class DetermineColumnNameContainingHumanText(BaseModel):
        """Given a list of column names and values, deterimine the column name containing text from the user only"""
        text_column: Literal[tuple(df.columns.to_list())]
    
    model_with_struct_output = init_model(response_format=DetermineColumnNameContainingHumanText)

    message_content = str(df.columns) + "\n\n values: " + str(df.values[0])

    response = model_with_struct_output.invoke(message_content)

    column_name = response.text_column

    loader = CSVLoader(file_path, content_columns=f"{column_name}")
    docs = loader.load()
    for doc in docs:
        
        original_filename = Path(file_path).stem + Path(file_path).suffix 
        doc.page_content = doc.page_content.lstrip(f"{column_name}: ")
        filename_uuid5 = str(uuid5(NAMESPACE_URL, original_filename))
        id = str(uuid4())
        metadata_update = {"id":id, "filename": original_filename, "filename_uuid5":filename_uuid5}
        doc.metadata.update(metadata_update)
    
    return docs
    
    