from typing import Any

import logging

logger = logging.getLogger(__name__)

from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.anubis.utils.model import init_model

logger = logging.getLogger(__name__)


from pydantic.dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from langchain_core.messages.utils import count_tokens_approximately

from uuid import uuid4, uuid5, NAMESPACE_URL

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from typing import Literal
from pydantic import Field
from src.anubis.utils.analysis.analysis_methods import perform_ocean_analysis

from src.anubis.utils.schema import (
    TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS,
    MONOLOGUE_PRESENTATION_OR_SERIES_OF_QUOTES,
    REFERENCE_DOCUMENT_OR_BIOGRAPHICAL_CONVERSATIONAL_INFORMATION,
)
from src.anubis.utils.schema import (
    TextualSituationalAwareness,
    MonologuePresentationOrSeriesOfQuotes,
    ReferenceDocumentOrBiographicalConversationalInformation,
)

from langchain_core.messages import SystemMessage, HumanMessage


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def normal_chunking(
    text_content: str, metadata: dict, chunk_size: int = 1024, chunk_overlap: int = 150
):
    # 1. Define the hierarchy of separators
    # The order matters: it tries to split at the top of the list first.
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    # 2. Initialize the splitter
    # Since you are moving to a new embedding model,
    # using a token-based length function is safer than character count.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=count_tokens_approximately,  # Or count_tokens_approximately if you prefer
        is_separator_regex=False,
    )

    # 3. Create the Documents
    # create_documents handles the string-to-Document conversion and metadata propagation
    texts = text_content if type(text_content) == list else text_content.splitlines()
    docs = text_splitter.create_documents(texts=texts, metadatas=[metadata])

    return docs


# Process TEXT TO DOCUMENT
async def process_text_to_document(
    metadata, user_id, assistant_id, media_item
) -> list[Document]:
    """Process text to document;
    document chunking necessity,
    situation determination,
    and future use of the data (vectorstore, analysis, adapter) are handled here
    """
    logger.info(f"Handling text in process media")

    proprietary_content = metadata.get("proprietary_content", False)

    proprietary_content_classification_model = init_model(
        model_without_tools=False,
        response_format=ReferenceDocumentOrBiographicalConversationalInformation,
    )
    text_content = media_item.get("content", "")

    classification = await proprietary_content_classification_model.ainvoke(
        [
            SystemMessage(
                content=REFERENCE_DOCUMENT_OR_BIOGRAPHICAL_CONVERSATIONAL_INFORMATION
            ),
            HumanMessage(content=text_content[:5000]),
        ]
    )
    # if proprietary_content:
    # TODO: CALCULATE TOKEN USAGE response['response_metadata']

    if classification.is_menu_or_religious_text:
        logger.warning(
            f"proprietary content: No single target; media is only uploaded to vectorstore"
        )

        classification_metadata = {
            "classified_situation": "proprietary content",
            "classification_reasoning": classification.reasoning,
        }

        documents = await process_text_media_item_target_for_vectorstore(
            media_item=media_item,
            user_id=user_id,
            assistant_id=assistant_id,
            classification_metadata=classification_metadata,
            use_semantic_chunks=False,
        )

        for document in documents:
            document.metadata.update({"vectorstore_acceptable": True})
        return documents

    else:
        logger.info(
            f"There is a target and the content of the text needs to be analyzed (is this a monologue or multi-speaker or strictly Q & A; how many speakers, etc.)"
        )

        # Analyze text situation

        model_with_structured_output = init_model(
            model_without_tools=False, response_format=TextualSituationalAwareness
        )

        system_prompt = TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS

        classification = await model_with_structured_output.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content},
            ]
        )
        logger.info(f"Situation classification: {classification.classified_situation}")

        logger.info(f"Reason for classification : {classification.reasoning}")

        classification_metadata = {
            "classified_situation": classification.classified_situation,
            "classification_reasoning": classification.reasoning,
        }
        if classification.classified_situation == "single_speaker":
            # TODO: Determine if the text is of the single person directly speaking as a quote for the entire document in the media item

            # is this content written in first person or is this content about the individual?
            # from src.anubis.utils.prompts.system_prompts import DETERMINE_TEXT_SINGLE_SPEAKER_FIRST_PERSON_TONE_OF_VOICE_SYSTEM_PROMPT

            monologue_vs_distinct_quotes_classification_model = init_model(
                model_without_tools=False,
                response_format=MonologuePresentationOrSeriesOfQuotes,
            )

            system_prompt = MONOLOGUE_PRESENTATION_OR_SERIES_OF_QUOTES
            input = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text_content),
            ]

            # TODO: CALCULATE TOKEN USAGE response['response_metadata']

            response = await monologue_vs_distinct_quotes_classification_model.ainvoke(
                input
            )

            classification_metadata = {
                "classified_situation": response.classified_situation,
                "classification_reasoning": response.reason,
            }
            if response.classified_situation == "SeriesOfDistinctQuotes":
                contiguous_lines = media_item.get("content", "")
                lines = contiguous_lines.splitlines()
                idx = 0
                all_documents = []
                for line in lines:
                    # Extract text content
                    media_type = media_item.get("item", "")
                    text_content = line
                    filename = media_item.get("metadata", {}).get("filename", "")
                    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename))

                    source_metadata = media_item.get("metadata", {})
                    source = source_metadata.get("source", "user_upload")

                    if not text_content or (text_content == ""):
                        logger.warning("Empty text content in media_item")
                        continue

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
                            "filename": filename,
                            "document_id": str(uuid4()),
                            "filename_uuid5": filename_uuid5,
                            "namespace": "quote",
                        },
                    )
                    idx += 1

                    if classification_metadata is not None:
                        doc.metadata.update(classification_metadata)
                        doc.metadata.update({"vectorstore_acceptable": True})
                    docs = [doc]
                    all_documents.extend(docs)
                [
                    document.metadata.update({"total_chunks": idx})
                    for document in all_documents
                ]

                additional_metadata = {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": current_timestamp,
                    "source": source,
                    "type": "text",
                    "filename": filename,
                    "filename_uuid5": filename_uuid5,
                    "namespace": "identity",
                    "analysis_acceptable": True,
                }
                analysis_documents = await perform_ocean_analysis(
                    human_message=HumanMessage(content=media_item.get("content")),
                    additional_metadata=additional_metadata,
                )

                all_documents.extend(analysis_documents)
                # Analysis Acceptable to be determined here on bulk of media
                return all_documents

            else:
                # Handle Monologue or Presentation
                logger.info("Monologue or Presentation detected")
                # @dataclass
                # class DetermineTextFirstPersonToneOfVoice(BaseModel):
                #     """  """
                #     classification: Literal["first_person_directly_speaking", "content_about_target_NOT_the_target_directly_speaking"]
                #     reason: str = Field()
                # classification_model = init_model(response_format=DetermineTextFirstPersonToneOfVoice)

                # model_with_structured_output_classify_text_perspective = init_model()
                # input = [{"role": "system", "content": DETERMINE_TEXT_SINGLE_SPEAKER_FIRST_PERSON_TONE_OF_VOICE_SYSTEM_PROMPT}, {"role": "user", "content": media_item['content']}]
                # response = model_with_structured_output_classify_text_perspective.ainvoke(input=input)
                #
                # if response['classification'] == "first_person_directly_speaking":
                # Direct quote content of only the target speaker speaking in the entire media item.
                # """ IF THE DETERMINATION ABOVE IS TRUE """
                # TODO: format for Adapter: generate a prompt to the single speaker monologue; create q & a format; create document
                # TODO: GENERATE A PROMPTING QUESTION AND CREATE A TRAINING DOCUMENT WITH BOTH GENERATED QUESTION AND THIS RESPONSE
                #  """""""""
                # TODO: format for Baseline ground truth using only text from first-person perspective of the target speaker (ultimately combine with analysis for evaluation; is this generated text something the target speaker would say/know; is this how they behave, their internal decision tree chain-of-thought, are their emotions and emotional sentiment in alignment given ground truth primary resource experiences? (include vader sentiment of baseline ground truth))
                # BASELINE CODE BELOW
                # make_pg_store
                # namespace = (user_id, assistant_id)
                # baseline_evaluation_quote_data = aget(namespace, key="baseline_evaluation_quote_data")
                # metadata = baseline_evaluation_quote_data.metadata
                # metadata_update = {"baseline_evaluation_quote_data": {"uuid4()":{"data": "unchunked direct_quote from media_item['content']", "metadata":{"created_at":"", "filename":"", "user_id":"", "assistant_id":""}}}} # also the structure of original metadata
                # metadata.update(metadata_update)
                #
                # update the metadata object with an overwrite
                # aput(namespace, key="baseline_evaluation_quote_data", value=metadata) # this extends the metadata dictionary

                # iterate through the uuid4's to pull the "data" and have all the orginal quote content for evaluation of AI responses.
                # """ REGARDLESS OF DETERMINATION (accept both facts and direct quotes) """
                # format for vectorstore: chunk and upload to vectorstore
                logger.info(
                    f"proccess_text_media_item_target_for_vectorstore BREAKPOINT in Process media item task: type = text"
                )
                documents = await process_text_media_item_target_for_vectorstore(
                    media_item=media_item,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    classification_metadata=classification_metadata,
                    use_semantic_chunks=False,
                )
                for document in documents:
                    document.metadata.update({"vectorstore_acceptable": True})
            # TODO: format for analysis: analyze for content about the target
            # USE THE DETERMINATION TO AUGMENT ANALYSIS AND NOTE THAT THE CONTENT IS EITHER ABOUT THE TARGET OR IS FROM THE TARGET SPEAKING DIRECTLY
        elif classification["classified_situation"] == "q_and_a_dialogue":
            logger.warning(f"Q & A DIALOGUE CLASSIFICATION DETECTED")
            logger.warning(
                f"""
            # TODO: format for vectorstore; CHUNKS OF NON-TARGET CANNOT BE IN THE VECTORSTORE WITHOUT THE TARGET SPEAKER
                # documents = process_text_media_item_target_for_vectorstore(
                #     media_item, 
                #     user_id=user_id, 
                #     assistant_id=assistant_id, 
                #     classification_metadata=classification_metadata
                # )
                # for document in documents: 
                #     document.metadata.update({"formatted_type": "vectorstore"})
                    
            # TODO: format for analysis: extract target speaker only and analyze with llm
            # TODO: format for Adapter: Q & A format document
            """
            )
        elif classification["classified_situation"] == "multi_speaker":
            logger.warning(f"MULTI-SPEAKER CLASSIFICAITON DETECTED")
            logger.warning(
                f"""
            # TODO: format for vectorstore; CHUNKS OF NON-TARGET CANNOT BE IN THE VECTORSTORE WITHOUT THE TARGET SPEAKER; CONTENT IS NOT DIALOGUE; SPEAKER NEEDS TO BE IDENTIFIED IN THE TEXT
            # TODO: format for analysis: TARGET MUST BE IDENTIFIED, LLM IS USED TO ANALYZE CONTENT ABOUT THE TARGET ONLY
            # TODO: format for Adapter: ALL OTHER NON-TARGET SPEAKERS ARE CLASSIFIED AS USER AND THE TARGET SPEAKER IS CLASSIFIED AS AI FOR THE FORMAT.
            """
            )
        elif classification["classified_situation"] == "other":
            logger.warning(
                f"OTHER text situation classification detected. Inspect and handle the situation appropriately. Currently handled in the same procedure as proprietary content."
            )

            logger.warning(
                f"proprietary content procedure: No single target; media is only uploaded to vectorstore"
            )

            documents = await process_text_media_item_target_for_vectorstore(
                media_item=media_item,
                user_id=user_id,
                assistant_id=assistant_id,
                classification_metadata=classification_metadata,
                use_semantic_chunks=True,
            )
            for document in documents:
                document.metadata.update({"formatted_type": "vectorstore"})
        else:
            logger.warning(
                f"Error: situation classification is not of type other, multi_speaker, q_and_a_dialogue, or single_speaker"
            )
        return documents


# TEXT TO VECTORSTORE

"""
Text chunking module for processing large text files into Documents
for LangGraph-MongoDB vectorstore with harrier-oss-v1-270m embeddings (384 dimensions)
"""

from pydantic import BaseModel


async def split_text_into_chunks(
    text_splitter: RecursiveCharacterTextSplitter,
    text_content: str,
    source_metadata: dict,
    source: str,
    user_id: str,
    assistant_id: str,
    classification_metadata: dict,
    idx: int,
    document_id: str,
    filename: str,
    filename_uuid5: str,
):
    """TEXT CHUNKING HELPER FUNCTION: process_text_media_item_target_for_vectorstore"""

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
                "filename": filename,
                "filename_uuid5": str(uuid5(NAMESPACE_URL, filename)),
                "document_id": str(uuid4()),
            },
        )
        idx += 1
        doc.metadata.update(source_metadata)
        if classification_metadata is not None:
            doc.metadata.update(classification_metadata)
        documents.append(doc)

    return idx, documents


async def process_text_media_item_target_for_vectorstore(
    media_item: Dict[str, Any],
    user_id: str,
    assistant_id: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
    classification_metadata: Optional[dict] = None,
    use_semantic_chunks: bool = False,
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
                   Optimized for harrier-oss-v1-270m (384 dimensions)
        chunk_overlap: Number of overlapping characters between chunks (default: 50)
        separators: Custom separators for text splitting (optional)

    Returns:
        List of Document objects ready for vectorstore upload
    """

    logger.warning(f"PROCESS_TEXT_MEDIA_ITEM ENTRYPOINT")
    try:

        # Extract text content
        media_type = media_item.get("item", "")
        text_content = media_item.get("content", "")
        filename = media_item.get("metadata", {}).get("filename", "")
        filename_uuid5 = str(uuid5(NAMESPACE_URL, filename))

        source_metadata = media_item.get("metadata", {})
        source = source_metadata.get("source", "user_upload")

        if not text_content:
            logger.warning("Empty text content in media_item")
            return []

        # NOTE: semantic chunking is intentionally disabled; all text chunking uses normal_chunking.
        _ = use_semantic_chunks
        all_documents = normal_chunking(
            text_content=text_content,
            metadata={},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        current_timestamp = datetime.now(tz=timezone.utc).isoformat()
        total_chunks = len(all_documents)
        idx = 0

        for document in all_documents:
            document.metadata.update(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": current_timestamp,
                    "processing_task_id": str(uuid4()),
                    "source": source,
                    "type": "text",
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "filename": filename,
                    "filename_uuid5": filename_uuid5,
                    "document_id": str(uuid4()),
                }
            )
            document.metadata.update(source_metadata)
            if classification_metadata is not None:
                document.metadata.update(classification_metadata)
            idx += 1

        # Update the total chunks in the metadata:
        _ = [
            document.metadata.update({"total_chunks": idx})
            for document in all_documents
        ]
        return all_documents

    except Exception as e:
        logger.exception(
            f"Error in text chunking during process media item target for vector store: {e}"
        )
        raise
