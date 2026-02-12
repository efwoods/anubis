from typing import Any

import logging
logger = logging.getLogger(__name__)


# TEXT TO VECTORSTORE

"""
Text chunking module for processing large text files into Documents
for LangGraph-MongoDB vectorstore with all-MiniLM-L6-v2 embeddings (384 dimensions)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.anubis.utils.configuration import GlobalConfiguration
from src.anubis.utils.model import init_model

logger = logging.getLogger(__name__)


from pydantic.dataclasses import dataclass

from sentence_transformers import SentenceTransformer

# Load once — reuse
configuration = GlobalConfiguration()
embedder = SentenceTransformer(configuration.embedding_model)
embedding_tokenizer = embedder.tokenizer

def token_length(text: str) -> int:
    # Most accurate: no special tokens, just like sentence-transformers pooling
    return len(embedding_tokenizer.encode(text, add_special_tokens=False))
@dataclass
class SemanticChunkIndexList:
    index: list[int]

async def split_text_into_chunks(
        text_splitter: RecursiveCharacterTextSplitter, 
        text_content: str, 
        source_metadata: dict,
        source: str, 
        user_id: str, 
        assistant_id: str,
        classification_metadata: dict,
        idx: int, 
    ):
    
    """ TEXT CHUNKING HELPER FUNCTION: process_text_media_item_target_for_vectorstore """

    logger.info(f"SPLIT TEXT INTO CHUNKS ENTRYPOINT")

    # Split text into chunks
    text_chunks = text_splitter.split_text(text_content)
    
    logger.info(f"Split text into {len(text_chunks)} chunks")
    
    # Create Document objects for each chunk
    documents = []
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()
    
    for chunk in text_chunks:
        
        doc = Document(
            page_content=chunk,
            metadata={
                "user_id": user_id,
                "assistant_id": assistant_id,
                "created_at": current_timestamp,
                "processing_task_id": str(uuid4()),
                "source": source,
                "type": "text",
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                # Include any additional metadata from original media_item
                **{k: v for k, v in source_metadata.items() if k not in ["source"]}
            }
        )
        idx += 1
        if classification_metadata is not None:
            doc.metadata.update(classification_metadata)
        documents.append(doc)
    
    return idx, documents

async def process_text_media_item_target_for_vectorstore(
    media_item: Dict[str, Any],
    user_id: str,
    assistant_id: str,
    configuration: GlobalConfiguration,
    chunk_size: int = 256,
    chunk_overlap: int = 25,
    separators: Optional[List[str]] = None,
    classification_metadata: Optional[dict] = None,
    use_semantic_chunks: bool = True
) -> List[Document]:
    """
    USAGE README: 
    
    THIS CHUNKS TEXT INTO DOCUMENTS FOR THE VECTORSTORE. 
    IT IS EXPECTED THAT THE MEDIA ITEM TEXT IS OF AN IDENTIFIED TARGET ONLY BEFORE PROCESSING WITH THIS FUNCTION.
    USED WHEN THERE IS ONLY A SINGLE TARGET OR THE TEXT IS NON-DIALOGUE AND NON-MULTISPEAKER (MEDIA ARTICLES, ETC.)

    Process text content using recursive text splitter and convert to Documents.
    
    This function handles large text files by:
    1. Extracting text from media_item
    2. Splitting text into chunks using RecursiveCharacterTextSplitter
    3. Creating Document objects with proper metadata for each chunk
    
    Args:
        media_item: Dictionary containing text data with 'text' key
        user_id: User identifier for metadata
        assistant_id: Assistant identifier for metadata
        chunk_size: Maximum size of each text chunk (default: 500)
                   Optimized for all-MiniLM-L6-v2 (384 dimensions)
        chunk_overlap: Number of overlapping characters between chunks (default: 50)
        separators: Custom separators for text splitting (optional)
    
    Returns:
        List of Document objects ready for vectorstore upload
    """
    
    logger.warning(f"PROCESS_TEXT_MEDIA_ITEM ENTRYPOINT")
    try:

        # Extract text content
        text_content = media_item.get("content", "")

        source_metadata = media_item.get("metadata", {})
        source = source_metadata.get("source", "user_upload")

        if not text_content:
            logger.warning("Empty text content in media_item")
            return []

        # Default separators optimized for semantic coherence
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "? ",    # Question endings
                "! ",    # Exclamation endings
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Spaces
                ""       # Characters
            ]

        token_text_content_length = token_length(text_content)    
        if token_text_content_length > chunk_size and use_semantic_chunks:
        
            # Define meaningful chunks first

            tools = []

            model_structured_output = init_model(
                        configuration.provider_model,
                        configuration.llama_api_base_url,
                        configuration.llama_api_key,
                        tools,
                        configuration.dev,
                        response_format=SemanticChunkIndexList
            )

            SEMANTICALLY_CHUNK_TEXT_SYSTEM_PROMPT = """ 
            <Role>
            You are an expert at dividing text into semantically coherent sections.
            </Role>

            <Instructions> 
            Given the text below, return ONLY a JSON list of integers representing the starting character indices where each new semantic chunk should begin.

            - Determine the number of semantically meaningful chunks. Use separators, paragraph tabs, and indicators along with topic changes to determine the number of semantically meaningful chunks.
            - Reasonable sections to split the text include distinct topics. 
            - You are returning a list of integers
            - Return a list of the text portions. 
            - Attempt to create lists with the number of tokens less than {chunk_size} tokens with approximately 50  70 tokens in 280 characters if possible otherwise maintain the semantically meaningful chunk.
            - The length of the number of characters in each chunk takes secondary precedence to chunks that are semantically meaningful and is an OPTIONAL requirement.
            - Reasonable sections to split text are when there is a topic change.
            - Chunks must be contiguous and cover the entire text without overlap or gap.
            - Do not skip or repeat any content.
            - Return exact original substrings when sliced with these indices.
            - Do not paraphrase or change any wording.
            - Return format: [0, 142, 378, 915, ...]  (last index may be len(text))
            - Chunk boundaries MUST occur only immediately after one of these separators:
              - "\\n\\n"
              - "\\n"
              - ". "
              - "? "
              - "! "
              - "; "
              - ", "
              - " "
            - Never split inside a word, number, quote, or mid-sentence.
            - All chunks must contain complete thoughts/sentences when possible.
            - Prefer semantic coherence (topic shift) over length.
            </Instructions> 

            <Rules>
            - all chunks MUST contain complete sentences. Do dont start or end a chunk in the middle of a word
            - Do not change any of the original text.
            - Return all text that was originally sent.
            - If there are no meaningful sections or portions, then DO NOT separate the text and only return a list containing a single item of the original text.
            - Always return a list.
            - Attempt to create lists with the number of characters less than {chunk_size} tokens with approximately 50 to 70 tokens in 280 characters if possible otherwise maintain the semantically meaningful chunk.
            - The length of the number of characters in each chunk takes secondary precedence to chunks that are semantically meaningful and is an OPTIONAL requirement.
            - Chunks must be contiguous and cover the entire text without overlap or gap.
            - Do not skip or repeat any content.
            - Return exact original substrings when sliced with these indices.
            - Do not paraphrase or change any wording.
            - Return format: [0, 142, 378, 915, ...]  (last index may be len(text))
            - Chunk boundaries MUST occur only immediately after one of these separators:
              - "\\n\\n"
              - "\\n"
              - ". "
              - "? "
              - "! "
              - "; "
              - ", "
              - " "
            - Never split inside a word, number, quote, or mid-sentence.
            - All chunks must contain complete thoughts/sentences when possible.
            - Prefer semantic coherence (topic shift) over length.

            </Rules>

            """

            logger.warning(f"THE CHUNKS ARE SEMANTICALLY MEANINGFUL BUT ARE LONGER THAN 500 CHARACTERS AND CONTAIN PARTS OF WORDS. THESE CHUNKS WILL BE CHUNKED AGAIN. THE INITIAL CHUNKING IS SEMANTICALLY MEANINGFUL")

            formatted_system_prompt = SEMANTICALLY_CHUNK_TEXT_SYSTEM_PROMPT.format(chunk_size=chunk_size)

            model_result = await model_structured_output.ainvoke(input=[
                {"role": "system", "content": formatted_system_prompt}, 
                {"role": "user", "content": text_content}
            ])

            """ Parse semantic chunks using start and end indicators """

            try:
                starts = model_result.get("index", [])
                assert isinstance(starts, list) and all(isinstance(i, int) for i in starts)
            except:
                raise ValueError("LLM did not return value index list")

            starts = sorted(set(starts)) # ensure the list of integers is sorted
            if starts[0] != 0:
                starts = [0] + starts

            chunks = []
            for i in range(len(starts)):
                start = starts[i]
                end = starts[i+1] if i+1 < len(starts) else len(text_content)
                chunk = text_content[start:end]
                chunks.append(chunk)


            logger.info(f"model_result breakpoint on semantic chunking of text: {chunks}")

            # Initialize recursive text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=token_length,
                is_separator_regex=False,
            )

            if isinstance(chunks, list):
                # Ensure the model has not altered the text in any way (length of text is the same before and after invocation)
                text_total_text_length_of_semantic_chunks = 0 

                for chunk in chunks:
                    text_total_text_length_of_semantic_chunks += len(chunk)

                if text_total_text_length_of_semantic_chunks == text_content.__len__():
                    # Create Document objects for each chunk
                    all_documents = []
                    current_timestamp = datetime.now(tz=timezone.utc).isoformat()
                    idx = 0

                    for semantic_text_chunk in chunks:
                        # Determine the length in tokens per semantic text chunk
                        semantic_text_chunk_token_size = token_length(semantic_text_chunk)

                        if semantic_text_chunk_token_size < chunk_size:
                            """ create documents of semantic chunks """


                            doc = Document(
                                page_content=semantic_text_chunk,
                                metadata={
                                    "user_id": user_id,
                                    "assistant_id": assistant_id,
                                    "created_at": current_timestamp,
                                    "processing_task_id": str(uuid4()),
                                    "source": source,
                                    "type": "text",
                                    "chunk_index": idx,
                                    # Include any additional metadata from original media_item
                                    **{k: v for k, v in source_metadata.items() if k not in ["source"]}
                                }
                            )
                            idx += 1
                            if classification_metadata is not None:
                                doc.metadata.update(classification_metadata)
                        else: 
                            # Split the semantic chunk into smaller chunks
                            idx, documents = await split_text_into_chunks(
                                text_splitter=text_splitter, 
                                text_content=semantic_text_chunk,
                                source_metadata=source_metadata,
                                source=source,
                                user_id=user_id,
                                assistant_id=assistant_id,
                                classification_metadata=classification_metadata,
                                idx=idx,
                            )
                            if not isinstance(documents, list): # Redundant type verification
                                documents = [documents] 
                            all_documents.extend(documents)
                else:
                    logger.warning(f"Warning: Text content is not the same length as the total text length of the semantically meaninful chunks. Text has been altered. Chunking original text.")
                    all_documents = []
                    current_timestamp = datetime.now(tz=timezone.utc).isoformat()
                    idx = 0
                    idx, documents = await split_text_into_chunks(
                                text_splitter=text_splitter, 
                                text_content=text_content,
                                source_metadata=source_metadata,
                                source=source,
                                user_id=user_id,
                                assistant_id=assistant_id,
                                classification_metadata=classification_metadata,
                                idx=idx,
                            )
                    if not isinstance(documents, list): # Redundant type verification
                        documents = [documents] 
                    all_documents.extend(documents)
            else: 
                logger.warning(f"Error: semantic model chunking result is not a list during 'process_text_media_item_target_for_vectorstore'. Chunking Original Text content.")
                all_documents = []
                current_timestamp = datetime.now(tz=timezone.utc).isoformat()
                idx = 0
                idx, documents = await split_text_into_chunks(
                            text_splitter=text_splitter, 
                            text_content=text_content,
                            source_metadata=source_metadata,
                            source=source,
                            user_id=user_id,
                            assistant_id=assistant_id,
                            classification_metadata=classification_metadata,
                            idx=idx,
                        )
                if not isinstance(documents, list): # Redundant type verification
                    documents = [documents] 
                all_documents.extend(documents)

        else:
            if token_text_content_length > chunk_size:
                logger.info(f"length of text context is greater than the chunk size therefore use semantic chunks was set to FALSE; CHUNKING TEXT NON-SEMANTICALLY")


                # Initialize recursive text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators,
                    length_function=len,
                    is_separator_regex=False,
                )

                all_documents = []
                current_timestamp = datetime.now(tz=timezone.utc).isoformat()
                idx = 0
                idx, documents = await split_text_into_chunks(
                            text_splitter=text_splitter, 
                            text_content=text_content,
                            source_metadata=source_metadata,
                            source=source,
                            user_id=user_id,
                            assistant_id=assistant_id,
                            classification_metadata=classification_metadata,
                            idx=idx,
                        )
                if not isinstance(documents, list): # Redundant type verification
                    documents = [documents] 
                all_documents.extend(documents)

            else:
                logger.info(f"use_semantic_chunks is False AND token_text_content_length is less than or equal to the chunk_size (256 tokens before truncation)")
                idx = 0
                text_chunk_token_size = token_length(text_content)
                all_documents = []

                # Redundant identification of chunk_size
                if text_chunk_token_size <= chunk_size:
                    """ create document of single text chunk """
                    current_timestamp = datetime.now(tz=timezone.utc).isoformat()
    
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "created_at": current_timestamp,
                            "processing_task_id": str(uuid4()),
                            "source": source,
                            "type": "text",
                            "chunk_index": idx,
                            # Include any additional metadata from original media_item
                            **{k: v for k, v in source_metadata.items() if k not in ["source"]}
                        }
                    )
                    idx += 1

                    if classification_metadata is not None:
                        doc.metadata.update(classification_metadata)
                    docs = [doc]
                    all_documents.extend(docs)

        # Update the total chunks in the metadata:
        result_of_document_metadata_update = [document.metadata.update({"total_chunks": idx}) for document in all_documents]

        return all_documents

    except Exception as e:
        logger.error("Error in text chunking during process media item target for vector store: {e}")
        raise e