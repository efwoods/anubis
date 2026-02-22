# Nodes for Identifying and Handling each media type 

from typing import Dict
from datetime import datetime, timezone
from langchain_core.documents import Document
from typing import Dict, Any
from uuid import uuid4

import logging
logger = logging.getLogger(__name__)
import base64

# At top of file
from langchain_core.documents import Document
import tempfile

import base64
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime

# from langgraph.config import get_store

from langgraph.store.base import BaseStore

from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store

from src.subgraphs.process_media_graph.utils.helper_functions import process_text_media_item_target_for_vectorstore

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.anubis.utils.model import init_model

from langgraph.prebuilt import ToolRuntime
from langchain.tools import tool

from src.anubis.utils.configuration import GlobalConfiguration

from langchain_core.runnables import RunnableConfig

async def process_uploaded_files_and_label_media_type(
    state: GlobalState, 
    runtime: Runtime[GlobalContext], 
    config: RunnableConfig,
    store: BaseStore
) -> Dict[str, Any]:
    """
    Convert FastAPI UploadFile objects into standardized media format.
    This is the entry point for direct file uploads (not from messages).
    """
    
    logger.info(f"Process uploaded files NODE")

    if isinstance(runtime.context.user_ctx, dict):
        user_id = runtime.context.user_ctx.get("user_id", "")
    else:
        user_id = getattr(runtime.context.user_ctx, "user_id", "")

    if isinstance(runtime.context.assistant_ctx, dict):
        assistant_id = runtime.context.assistant_ctx.get("assistant_id", "")
    else:
        assistant_id = getattr(runtime.context.assistant_ctx, "assistant_id", "")

    # logger.info("STORE ACCESS TESTING")
    # namespace = ("evan")
    # await store.aput(namespace=namespace, key="evan", value={"name": "evan"})
    # get_value = await store.aget("evan", key="name")
    # logger.info("get_value: {get_value}")


    media_files = state.get('media_files', [])
    
    if not media_files:
        logger.info("No media files to process")
        return {"media_list": []}
    
    logger.info(f"Processing {len(media_files)} uploaded files")
    
    media_list = []
    
    for file_data in media_files:
        try:
            # Extract file info
            filename = file_data.get('filename', 'unknown')
            content_type = file_data.get('content_type', '')
            file_bytes = file_data.get('content')  # Raw bytes
            user_id = file_data.get("user_id")
            assistant_id = file_data.get("assistant_id")
            reference_image = file_data.get("reference_image")
            reference_audio = file_data.get("reference_audio")
            proprietary_content = file_data.get("proprietary_content") # This is a single body of text (scripture/menu/etc ... there is no single target, text only)
            
            logger.info(f"Processing file: {filename} ({content_type})")
            
            # Determine media type and convert to standardized format
            if content_type.startswith('image/'):
                # Convert image to base64
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "image",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "reference_image": reference_image
                    }
                })
            
            elif content_type.startswith('audio/'):
                # Handle audio files
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "audio",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "reference_audio": reference_audio                    
                    }
                })
            
            elif content_type.startswith('video/'):
                # Handle video files
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "video",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                    }
                })
            
            elif content_type in ['text/plain', 'application/json', 'text/markdown']:
                # Handle text files
                text_content = file_bytes.decode('utf-8')
                media_list.append({
                    "type": "text",
                    "content": text_content,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "proprietary_content": proprietary_content
                    }
                })
            
            elif content_type == 'application/pdf':
                # Handle PDFs
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "pdf",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "proprietary_content": proprietary_content
                    }
                })
            
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                continue
        
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    logger.info(f"Converted {len(media_list)} files to media format")
    
    return {
        "media_list": media_list,
        "media_files": []  # Clear after processing
    }

async def convert_media_list_to_text_document(state: GlobalState, runtime: Runtime[GlobalContext], store: BaseStore, config: RunnableConfig) -> Dict[str, Any]:
    """ 
    Media type in media list is determined at this point: 
    Convert the media in a list of one or more media to text in parallel.
    media items must have user_id and assistant_id as metadata.
    Exptected format:
    [
        {
            "type": "MEDIA_TYPE", 
            "data|text|indicator": "CONTENT OF MEDIA", 
            "metadata":{
                fields may include mime-type or the metadata may not exists at all
                }
        }, 
        ...
    ]
    I want to keep the media in a list and queue tasks for each item in the list 
    then I want to execute those tasks in parallel and update the final state 
    with the list of text Documents from the media:
    async def determine_media_type(state: GlobalState, context: GlobalContext, media_list: List[Dict]):
    """
    
    logging.info(f"DETERMINE_MEDIA_TYPE NODE")
    
    media_list = state.get('media_list', [])

    if not media_list:
        logger.info(f"No Meida to process")
        return {
            "media_list": []
        }

    logger.info(f"Processing {len(media_list)} media items")

    # Create tasks for parallel processing
    all_documents = []
    for media_item in media_list:
        docs = await process_media_item_task(media_item, runtime, config, store)
        
        for doc in docs:
            status = doc.metadata.get("status", "")
            if status == "error":
                error = doc.metadata.get("error", "")
                filename = doc.metadata.get("filename", "")
                logger.warning(f"Error processing media: {filename} {error}")
            else:
                all_documents.append(doc)

    # Identify Vector Store formatted documents
    # NOTE This contains only information from the target speaker or about the target speaker
    vector_store_document_list_formatted = [doc for doc in all_documents if doc.metadata["formatted_type"] == "vectorstore"]

    # # Analysis list (needs a node)
    # These documents have been formatted for analysis but have not yet been analyzed.
    # NOTE: Using non-target information will indicate triggers or responses. This information must not be lost. For analysis, keep both the User and other speakers but focus on the target.

    analysis_document_list_formatted = [doc for doc in all_documents if doc.metadata["formatted_type"] == "target_analysis"]

    # # Adapter list (needs a node)
    # documents_to_be_processed_for_adapter_training: List[Sequence[Document]] UPDATED RETURN VALUES IN RETURN processed into adapter training format and uploaded to storage

    adapter_document_list_formatted = [doc for doc in all_documents if doc.metadata["formatted_type"] == "adapter"]

    return {
        "vectorstore_documents_to_be_indexed": vector_store_document_list_formatted,
        "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant": analysis_document_list_formatted,
        "documents_to_be_processed_for_adapter_training": adapter_document_list_formatted,
        "media_list": [] # Clear processed media list in the state
    }

async def process_media_item_task(
    media_item: Dict[str, Any], 
    runtime: Runtime[GlobalContext], 
    config: RunnableConfig,
    store: BaseStore
) -> Document:
    """Task: Convert a single media item to a Document"""
    
    logger.info(f"process_media_item_task entry")

    media_type = media_item.get("type", "")
    
    metadata = media_item.get("metadata", {})

    user_id = metadata.get("user_id", "")
    assistant_id = metadata.get("assistant_id", "")

    logger.info(f"extracted user_id: {user_id}")
    logger.info(f"extracted assistant_id: {assistant_id}")

    filename = media_item['metadata']['filename']
    logger.info(f"Processing file: {filename}")

    configuration = runtime.context.configuration

    logger.info(f"Testing store access")

    namespace = ("testing","document")
    await store.aput(namespace=namespace, key="media", value={"media":media_item, "document":media_item['content']})
    testing_get = await store.aget(namespace=namespace, key="media")
    testing_search = await store.asearch(("testing", "document"), query="Shivon Zilis")
    logger.info(f"testing_get: {testing_get}")
    logger.info(f"get_value: {testing_search}")

    try:
        # Handle base64 images
        if media_type == "image":
            reference_image = media_item['metadata']["reference_image"]
            if "data" in media_item:
                # Base64 image
                image_data = media_item["data"]                
                
                    
                doc =  await extract_personality_from_image(image_data)
                    # Filter valid Documents and add metadata
                doc.metadata.update({
                    "user_id": user_id,
                    "assistant_id": assistant_id, 
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "reference_image": reference_image,
                    "filename": filename
                }) 

                if reference_image:
                    logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                    namespace=(user_id, assistant_id)
                    postgres_db_store = await make_pg_store(configuration)
                    metadata = doc.metadata
                    page_content = doc.metadata.get("page_content", "")
                    metadata.update({"page_content":page_content})

                    with postgres_db_store as pg_store:
                        pg_store.aput(
                            namespace, key="reference_image", 
                            value={"reference_image_data": image_data, "metadata": metadata})               
                docs = [doc]
                return docs
            
            elif "image_url" in media_item:
                # URL-based image
                url = media_item["image_url"].get("url", "")
                if url.startswith("data:image"):
                    # Extract base64 data
                    image_data = url.split(",", 1)[1]
                    
                    doc =  await extract_personality_from_image(image_data)
                    # Filter valid Documents and add metadata
                doc.metadata.update({
                    "user_id": user_id,
                    "assistant_id": assistant_id, 
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "reference_image": reference_image,
                    "filename": filename
                })  

                if reference_image:
                    logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                    namespace=(user_id, assistant_id)
                    postgres_db_store = await make_pg_store(configuration)
                    metadata = doc.metadata
                    page_content = doc.metadata.get("page_content", "")
                    metadata.update({"page_content":page_content})

                    with postgres_db_store as pg_store:
                        pg_store.aput(
                            namespace, key="reference_image", 
                            value={"reference_image_data": image_data, "metadata": metadata})                   
                docs = [doc]
                return docs
        
        # Handle text (Project Gutenberg; text files; list of media urls): https://claude.ai/chat/30c554c8-1386-4af2-9f19-f63b51942fc5
        elif media_type == "text":
            logger.info(f"Handling text in process media")
            
            proprietary_content = metadata.get("proprietary_content", False)
            
            if proprietary_content:
                logger.info(f"proprietary content: No single target; media is only uploaded to vectorstore")
                
                classification_metadata = {
                    "classified_situation": "proprietary content",
                    "classification_reasoning": "user_selected_classification_of_proprietary_content"
                }
                
                documents = await process_text_media_item_target_for_vectorstore(
                    media_item, 
                    user_id=user_id, 
                    assistant_id=assistant_id,
                    configuration=configuration,
                    classification_metadata=classification_metadata,
                    use_semantic_chunks=False
                )
                
                for document in documents: 
                            document.metadata.update({"formatted_type": "vectorstore"})

                return documents
            
            else:
                logger.info(f"There is a target and the content of the text needs to be analyzed (is this a monologue or multi-speaker or strictly Q & A; how many speakers, etc.)")
                
                # Analyze text situation
                from pydantic import BaseModel
                from pydantic.dataclasses import dataclass
                from typing import Literal
                from pydantic import Field
                @dataclass
                class TextualSituationalAwareness:
                    classified_situation: Literal["single_speaker", "q_and_a_dialogue", "multi_speaker", "other"]

                    reasoning: str = Field(
                        description = "Step-by-step reasoning behind the decision for the classified situation of the text. (single speaker monologue, single tweet from user, strictly Q & A, multi-speaker, Other)"
                    )

                tools = []

                model_with_structured_output = init_model(
                    configuration=configuration,
                    response_format=TextualSituationalAwareness
                )
                from src.anubis.utils.prompts.system_prompts import TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS

                system_prompt = TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS
                text_content = media_item.get("content", "")
                
                classification = await model_with_structured_output.ainvoke(input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_content}
                ])

                logger.info(f"Situation classification: {classification['classified_situation']}")

                logger.info(f"Reason for classification : {classification['reasoning']}")
                
                classification_metadata = {
                    "classified_situation": classification['classified_situation'], "classification_reasoning": classification['reasoning']
                }

                if classification['classified_situation'] == "single_speaker":

                    # TODO: Determine if the text is of the single person directly speaking as a quote for the entire document in the media item 
                    
                    # is this content written in first person or is this content about the individual?
                    # from src.anubis.utils.prompts.system_prompts import DETERMINE_TEXT_SINLGE_SPEAKER_FIRST_PERSON_TONE_OF_VOICE_SYSTEM_PROMPT
                    # @dataclass
                    # class DetermineTextFirstPersonToneOfVoice:
                    #
                    #   classification: Literal("first_person_directly_speaking", "content_about_target_NOT_the_target_directly_speaking")
                    #   reason: str = Field()
                    # model_with_structured_output_classify_text_perspective = init_model()
                    # input = [{"role": "system", "content": DETERMINE_TEXT_SINLGE_SPEAKER_FIRST_PERSON_TONE_OF_VOICE_SYSTEM_PROMPT}, {"role": "user", "content": media_item['content']}]
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
                        logger.info(f"proccess_text_media_item_target_for_vectorstore BREAKPOINT in Process media item task: type = text")
                        documents = await process_text_media_item_target_for_vectorstore(
                            media_item, 
                            user_id=user_id, 
                            assistant_id=assistant_id,
                            configuration=configuration,
                            classification_metadata=classification_metadata,
                            use_semantic_chunks=True
                        )
                        for document in documents: 
                            document.metadata.update({"formatted_type": "vectorstore"})

                    # TODO: format for analysis: analyze for content about the target
                    # USE THE DETERMINATION TO AUGMENT ANALYSIS AND NOTE THAT THE CONTENT IS EITHER ABOUT THE TARGET OR IS FROM THE TARGET SPEAKING DIRECTLY

                elif classification['classified_situation'] == "q_and_a_dialogue":
                    logger.warning(f"Q & A DIALOGUE CLASSIFICATION DETECTED")

                    logger.warning(f"""
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
                    """)

                elif classification['classified_situation'] == "multi_speaker":
                    logger.warning(f"MULTI-SPEAKER CLASSIFICAITON DETECTED")

                    logger.warning(f"""
                    # TODO: format for vectorstore; CHUNKS OF NON-TARGET CANNOT BE IN THE VECTORSTORE WITHOUT THE TARGET SPEAKER; CONTENT IS NOT DIALOGUE; SPEAKER NEEDS TO BE IDENTIFIED IN THE TEXT

                    # TODO: format for analysis: TARGET MUST BE IDENTIFIED, LLM IS USED TO ANALYZE CONTENT ABOUT THE TARGET ONLY

                    # TODO: format for Adapter: ALL OTHER NON-TARGET SPEAKERS ARE CLASSIFIED AS USER AND THE TARGET SPEAKER IS CLASSIFIED AS AI FOR THE FORMAT.
                    """)


                elif classification['classified_situation'] == "other":
                    logger.warning(f"OTHER text situation classification detected. Inspect and handle the situation appropriately. Currently handled in the same procedure as proprietary content.")
                    
                    logger.warning(f"proprietary content procedure: No single target; media is only uploaded to vectorstore")
                
                    documents = await process_text_media_item_target_for_vectorstore(
                        media_item, 
                        user_id=user_id, 
                        assistant_id=assistant_id,
                        configuration=configuration,
                        classification_metadata=classification_metadata,
                        use_semantic_chunks=True
                    )
                    for document in documents: 
                            document.metadata.update({"formatted_type": "vectorstore"})
                else:
                    logger.warning(f"Error: situation classification is not of type other, multi_speaker, q_and_a_dialogue, or single_speaker")

                return documents

        # Handle URLs
        elif media_type == "url":
            # TODO: Implement URL content fetching
            url = media_item.get("url", "")
            docs = [Document(
                page_content=f"Content from URL: {url}",
                metadata={"source": url, "type": "url", "status": "not_implemented"}
            )]
            return docs
           
        # Handle audio: https://claude.ai/chat/df5f518f-f846-4015-bb05-7adc6de96678
        elif media_type == "audio":
            """
            Detect the number of speakers
            Audio needs to be diarized if reference audio is available; 
            otherwise convert to text
            """
            reference_audio = media_item['metadata']['reference_audio']
            if "data" in media_item:
                # Base64 audio
                audio_data = media_item["data"]

                doc = await extract_text_from_audio(audio_data, configuration, user_id, assistant_id)

                # Add metadata
                doc.metadata.update({
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "type": "audio", 
                    "reference_audio": reference_audio,
                    "filename": filename
                })

                if reference_audio:
                    logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                    namespace=(user_id, assistant_id)
                    postgres_db_store = await make_pg_store(configuration)
                    metadata = doc.metadata
                    page_content = doc.metadata.get("page_content", "")
                    metadata.update({"page_content":page_content})

                    with postgres_db_store as pg_store:
                        pg_store.aput(
                            namespace, key="reference_audio", 
                            value={"reference_audio_data": audio_data, "metadata": metadata})                   

                docs = [doc]
                return docs
            
            elif "audio_url" in media_item:
                # URL-based audio
                url = media_item["audio_url"].get("url", "")
                if url.startswith("data:audio"):
                    # Extract base64 data
                    audio_data = url.split(",", 1)[1]

                    doc = await extract_text_from_audio(audio_data)

                    doc.metadata.update({
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "created_at": datetime.now(tz=timezone.utc).isoformat(),
                        "processing_task_id": str(uuid4()),
                        "type": "audio",
                        "reference_audio": reference_audio,
                        "filename": filename
                    })
                    if reference_audio:
                        logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                        namespace=(user_id, assistant_id)
                        postgres_db_store = await make_pg_store(configuration)
                        metadata = doc.metadata
                        page_content = doc.metadata.get("page_content", "")
                        metadata.update({"page_content":page_content})

                        with postgres_db_store as pg_store:
                            pg_store.aput(
                                namespace, key="reference_audio", 
                                value={
                                    "reference_audio_data": audio_data, 
                                    "metadata": metadata
                                })                   
                docs = [doc]
                return docs
        
        # Handle video
        elif media_type == "video":
            # TODO: Implement video processing
            docs = [Document(
                page_content="[Video processing not yet implemented]",
                metadata={"type": "video", "status": "not_implemented"}
            )]
            return docs
        
        else:
            logger.warning(f"Unsupported media type: {media_type}")
            docs = [Document(
                page_content=f"[Unsupported media type: {media_type}]",
                metadata={"type": media_type, "status": "unsupported"}
            )]
            return docs
    
    except Exception as e:
        # ERROR DOCUMENT
        logger.error(f"Error processing media item: {e}")
        documents =  [Document(
            page_content=f"[Error processing media: {str(e)}]",
            metadata={"type": media_type, "status": "error", "error": str(e)}
        )]
        return documents
    return await tool.ainvoke(media_item["content"])

async def extract_text_from_audio(audio_data: str, configuration: GlobalConfiguration, user_id: str, assistant_id: str) -> Document:
    """Extract text from audio using Hugging Face Whisper Large v3"""
    logger.info(f"needs reference audio from storage for speaker diarization (timestamps and who is speaking)")
    import base64
    import tempfile
    import os
    import asyncio
    import aiofiles

    logger.info(f"extract text from audio ENTRYPOINT")

    # store

    from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store

    pg_store_manager = await make_pg_store(configuration)

    async with pg_store_manager as pg_store:
        namespace = (user_id, assistant_id)
        reference_audio  = pg_store.aget(namespace=namespace, key="reference_audio")
        """
        {"reference_audio": {"reference_audio_data": data, "metadata": metadata}}
        """

    """ ANALYZE AUDIO """


    if reference_audio is not None:
        logger.info(f"Identify the number of speakers in the audio")
        logger.info(f"Identify if the target speaker is in the audio")
        logger.info(f"Diarize the audio (timestamps of who is speaking when)")
        
# Otherwise presume the audio is a single speaker of the target if there is no reference audio; mention in metadata
    if configuration.dev == "TRUE":
        from src.subgraphs.process_media_graph.utils.audio_transcription_local import get_whisper_pipeline
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)



            # Create temporary file in thread
            temp_file_directory, temp_audio_path = await asyncio.to_thread(
                tempfile.mkstemp,
                ".mp3"
            )

            # Write audio bytes asynchronously
            async with aiofiles.open(temp_audio_path, 'wb') as f:
                await f.write(audio_bytes)

                logger.info(f"Audio file written to {temp_audio_path}")
            try:
                # Get cached pipeline
                pipe = get_whisper_pipeline()

                # Run transcription in thread pool (it's CPU/GPU intensive)
                logger.info("Starting audio transcription...")
                result = await asyncio.to_thread(pipe, temp_audio_path)

                transcript = result["text"]

                # Create Document with transcription
                doc = Document(
                    page_content=transcript,
                    metadata={
                        "source": "audio_transcription",
                        "model": "whisper-large-v3",
                        "transcript_length": len(transcript),
                        "reference_audio_used": False,                }
                )
                return doc

            finally:
                # Clean up temporary file
                if temp_file_directory is not None:
                    await asyncio.to_thread(os.close, temp_file_directory)

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise

        finally:
            # Clean up temporary file in thread
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    await asyncio.to_thread(os.unlink, temp_audio_path)
                    logger.info(f"Cleaned up temporary file: {temp_audio_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_audio_path}: {cleanup_error}")
    else:
        logger.info(f"Production Audio Transcription Model Usage")
        # TODO: Complete Production Audio Transcription Model Usage"
        logger.error(f"Audio transcription failed production not implemented yet.")
        raise

async def extract_personality_from_image(
    image_data: str) -> Document:
    from src.anubis.utils.configuration import GlobalConfiguration
    """Extract personality description from image using vision LLM."""
    logger.info(f"needs reference image from storage for target identification (possibly object bounding box of the target)")
    # base64_image = self._image_to_base64(image_source)
    # base64_image = self.image_to_base64(image_path)

    logger.info(f"extract_personality_from_image entrypoint")    
    # Use reference image to help identify the person in the image.
    text_prompt_for_image_to_text_context = (
        "Describe the individual in the image in vivid detail using the FIRST PERSON PERSPECTIVE. "
        "Return only the description of the person using the FIRST PERSON PERSPECTIVE."
        "Do not mention that this is an image. "
        "Describe the qualities of the character of the person in full detail using the FIRST PERSON PERSPECTIVE and"
        "Describe the personality of this person so as to clearly visualize the person using the FIRST PERSON PERSPECTIVE."
        "Do describe the physical appearance using the FIRST PERSON PERSPECTIVE."
    )

    # these requests need to use the model in the graph rather than the requests because of 400 errors

    configuration = GlobalConfiguration()

    model = init_model(
        configuration=configuration
    )

    image_to_target_textual_description_payload = [
                        {
                            "type": "text",
                            "text": (text_prompt_for_image_to_text_context),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ]

    

    logger.info(message)

    response = await model.ainvoke([message])

    logger.info(f"response: {response}")

    if hasattr(response, 'content'):
        contextual_description = response.content
    else:
        contextual_description = str(response)

    logger.info(f"Extracted personality from image: {contextual_description[:100]}")

    return Document(
        page_content=contextual_description,
        metadata={
            "source": "vision_model", 
            "type": "personality_extraction", 
            "model": configuration.model
        }
    )

async def extract_media_from_message(state: GlobalState, runtime: Runtime[GlobalContext]):
    
    logger.info(f"Extract_media_from_message NODE")


    if isinstance(runtime.context.user_ctx, dict):
        user_id = runtime.context.user_ctx.get("user_id", "")
    else:
        user_id = getattr(runtime.context.user_ctx, "user_id", "")

    if isinstance(runtime.context.assistant_ctx, dict):
        assistant_id = runtime.context.assistant_ctx.get("assistant_id", "")
    else:
        assistant_id = getattr(runtime.context.assistant_ctx, "assistant_id", "")

    messages = state.get('messages', [])

    if not messages:
        logger.warning("No messages found in state")
        return {"media_list": []}
    
    logger.info(f"Processing {len(messages)} messages")

    # Get the most recent HumanMessage
    recent_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            recent_message = msg
            break
    
    if not recent_message:
        logger.info("No HumanMessage found")
        return {"media_list": []}

    content = recent_message.content

    # Handle string content (no media)
    if isinstance(content, str):
        logger.info("Message contains only text, no media")
        return {"media_list": []}
    
    # Handle list content (may contain media)
    if isinstance(content, list):
        logger.info(f"Message content has {len(content)} items")

        # Extract media (skip first item if it's text)
        media_list = []
        for item in content: 
            if isinstance(item, dict):
                item_type = item.get("type", "")

                # Skip pure text items
                if item_type == "text":
                    continue

                # Add media items
                if item_type in ["image", "image_url", "audio", "video", "url"]:
                    item["metadata"]['user_id'] = user_id
                    item["metadata"]['assistant_id'] = assistant_id
                    media_list.append(item)
                    # EACH ITEM NEEDS USER_ID AND ASSISTANT_ID FROM CONTEXT
                    # user_id
                    # assistant_id
            
        logger.info(f"Extracted {len(media_list)} media items")
        return {"media_list": media_list}

    logger.warning(f"Unexpected content type: {type(content)}")
    return {"media_list": []}
