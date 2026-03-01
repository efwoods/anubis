# Nodes for Identifying and Handling each media type 

from datetime import datetime, timezone
from langchain_core.documents import Document
from uuid import uuid4

import logging
logger = logging.getLogger(__name__)

import base64

import tempfile

import base64
from fastapi import HTTPException

from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.classes.configuration import GlobalConfiguration

logger = logging.getLogger(__name__)


from pydantic.dataclasses import dataclass

from langchain_core.messages.utils import count_tokens_approximately

from uuid import uuid4, uuid5, NAMESPACE_URL
from src.prompts.system_prompts import TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS

import logging
logger = logging.getLogger(__name__)

def init_model(configuration: GlobalConfiguration, 
               tools=[], response_format = None):

    model_name = configuration.model
    base_url = configuration.llama_api_base_url
    api_key = configuration.llama_api_key
    dev = configuration.dev

    logger.info(f"dev: {dev}")
    logger.info(f"base_url: {base_url}")
    logger.info(f"model_name: {model_name}")

    # if dev == 'TRUE':
    from langchain_openai import ChatOpenAI
    
    if response_format is None:
        model = ChatOpenAI(
                    model = model_name,
                    base_url = base_url,
                    temperature=0.1,
                    api_key = api_key,
                ).bind_tools(tools=tools)
    else: 
        model = ChatOpenAI(
            model = model_name,
            base_url = base_url,
            temperature=0.1,
            api_key = api_key,
        ).bind_tools(tools=tools)
        model = model.with_structured_output(response_format)
    # else: 
    #     from langchain_together import ChatTogether
    #     model = ChatTogether(model=model_name, temperature=0.1)
    return model


async def extract_user_id_assistant_id(config: Dict):
       
    user_id = config.get("configurable", {}).get("user_ctx", {}).get("user_id", "")
    if user_id == "":
        user_id = config.get("configurable",{}).get("user_id","")


    assistant_id = config.get("configurable", {}).get("assistant_ctx", {}).get("assistant_id", "")
    if assistant_id == "":
        assistant_id = config.get("configurable",{}).get("assistant_id","")

    # Ensure valid user_id and assistant_id
    if user_id == "" and (config.get("metadata", {}).get("from_studio") is True):
        user_id = "Anubis_from_studio_" + assistant_id
    user_id = "".join(user_id.strip())    
    assistant_id = "".join(assistant_id.strip())

    return user_id, assistant_id

async def process_uploaded_files_and_label_media_type(
    media_files: list, 
    config: Dict
) -> Dict[str, Any]:
    """
    Convert FastAPI UploadFile objects into standardized media format.
    This is the entry point for direct file uploads (not from messages).
    """
    
    logger.info(f"Process uploaded files NODE")
    user_id, assistant_id = await extract_user_id_assistant_id(config)

    # logger.info("STORE ACCESS TESTING")
    # namespace = ("evan")
    # await store.aput(namespace=namespace, key="evan", value={"name": "evan"})
    # get_value = await store.aget("evan", key="name")
    # logger.info("get_value: {get_value}")

    
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

    media_files = []
    
    return {
        "media_list": media_list,
    }

async def convert_media_list_to_text_document(
        media_list: list,
        config: Dict, 
        client: Any,
        ) -> Dict[str, Any]:
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
    
    if not media_list:
        logger.info(f"No Media to process")
        return {
            "media_list": []
        }

    logger.info(f"Processing {len(media_list)} media items")

    # Create tasks for parallel processing
    all_documents = []
    for media_item in media_list:
        docs = await process_media_item_task(media_item, config, client)
        
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
    }

import uuid

async def process_media_item_task(
    media_item: Dict[str, Any], 
    config: Dict,
    client: Any
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

    configuration = GlobalConfiguration()

    logger.info(f"Testing store access")

    # namespace = ("testing","document")
    # await store.aput(namespace=namespace, key="media", value={"media":media_item, "document":media_item['content']})
    # testing_get = await store.aget(namespace=namespace, key="media")
    # testing_search = await store.asearch(("testing", "document"), query="Shivon Zilis")
    # logger.info(f"testing_get: {testing_get}")
    # logger.info(f"get_value: {testing_search}")

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
                    "filename": filename,
                    "document_id": str(uuid4()),
                    "filename_uuid5":str(uuid.uuid5(uuid.NAMESPACE_URL, filename))
                }) 

                if reference_image:
                    logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                    namespace=(user_id, assistant_id, "reference_image")
                    document_json = document.to_json()
                    await client.store.aput(namespace, key=assistant_id, value={"reference_image_data": image_data, "document": document_json})
                    
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
                    "filename": filename,
                    "document_id": str(uuid4()),
                    "filename_uuid5":str(uuid.uuid5(uuid.NAMESPACE_URL, filename))
                })  

                if reference_image:
                    logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                    namespace=(user_id, assistant_id, "reference_image")
                    document_json = document.to_json()
                    await client.store.aput(namespace, key=assistant_id, value={"reference_image_data": image_data, "document": document_json})
                    
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

                model_with_structured_output = init_model(
                    configuration=configuration,
                    response_format=TextualSituationalAwareness
                )
                

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
                    "filename": filename,
                    "document_id": str(uuid4()),
                    "filename_uuid5":str(uuid.uuid5(uuid.NAMESPACE_URL, filename))
                })

                if reference_audio:
                    logger.warning(f"STORE REFERENCE AUDIO HERE: presuming upsert")
                    namespace=(user_id, assistant_id, "reference_audio")
                    await client.store.aput(namespace, key=assistant_id, value={"reference_audio_data": audio_data, "document": document.to_json()})                   

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
                        logger.warning(f"STORE REFERENCE AUDIO HERE: presuming upsert")
                        namespace=(user_id, assistant_id, "reference_audio")
                        metadata = doc.metadata
                        page_content = doc.metadata.get("page_content", "")
                        metadata.update({"page_content":page_content})

                        await client.store.aput(namespace, key=assistant_id,value={"reference_audio_data": audio_data, "metadata": metadata})                   

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

    # from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store

    # pg_store_manager = await make_pg_store(configuration)

    # async with pg_store_manager as pg_store:
    #     namespace = (user_id, assistant_id)
    #     reference_audio  = pg_store.aget(namespace=namespace, key="reference_audio")
    """
    {"reference_audio": {"reference_audio_data": data, "metadata": metadata}}
    """

    """ ANALYZE AUDIO """


    # if reference_audio is not None:
    logger.info(f"Identify the number of speakers in the audio")
    logger.info(f"Identify if the target speaker is in the audio")
    logger.info(f"Diarize the audio (timestamps of who is speaking when)")
        
# Otherwise presume the audio is a single speaker of the target if there is no reference audio; mention in metadata
    if configuration.dev == "TRUE":
        from api.src.classes.audio_transcription_local import get_whisper_pipeline
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

    input = {"messages": [{"role": "user", "content": image_to_target_textual_description_payload}]}
    

    response = await model.ainvoke(input=input)

    logger.info(f"response: {response}")

    if hasattr(response, 'content'):
        contextual_description = response.content
    else:
        contextual_description = str(response)

    logger.info(f"Extracted personality from image: {contextual_description[:100]}")

    return Document(
        page_content=contextual_description,
        metadata={
            "source": "vision_model_API", 
            "type": "personality_extraction", 
            "model": configuration.model
        }
    )

async def index_docs(
    vectorstore_documents_to_be_indexed: list, 
    config: Dict, 
    client: Any
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    logger.info(f"INDEXING DOCUMENTS")
    
    user_id, assistant_id = await extract_user_id_assistant_id(config)
   
    docs = vectorstore_documents_to_be_indexed
    
    filenames = [doc.metadata.get("filename") for doc in docs]
    try:
        assert(len(filenames) == len(docs))
    except AssertionError as e:
        logger.warning(f"Missing {len(docs) - len(filenames)} filenames on documents")
    
    if len(filenames) > 0:

        result = await batch_index_documents_vectorstore(client, user_id, assistant_id, docs, BATCH_SIZE=1000)

        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=f"Error processing media: {str(e)}"
            )


        logger.info(f"breaktpoint after batch_index_documents_vectorstore")

    return {"success": True, "number_of_documents_uploaded":result.get("documents_uploaded")}


from langgraph.store.base import PutOp

from uuid import uuid4

from src.classes.configuration import GlobalConfiguration

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
    
import asyncio

async def _put_with_retry(
    store,
    namespace: tuple,
    key: str,
    value: dict,
    max_retries: int = 2,
    retry_delay: float = 0.5,
):
    """Single put_item call with retry logic."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            await store.put_item(list(namespace), key=key, value=value)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                logger.warning(
                    f"put_item failed for key={key}, namespace={namespace} (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"put_item permanently failed for key={key}, namespace={namespace} after {max_retries + 1} attempts: {e}"
                )
                return {"success": False, "namespace":namespace, "key": key, "value":value}
    return {"namespace": namespace, "key": key, "value": value, "error": str(last_exc), "success": True}

async def _delete_with_retry(
    store,
    namespace: tuple,
    key: str,
    max_retries: int = 2,
    retry_delay: float = 0.5,
):
    """Single delete_item call with retry logic."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            await store.delete_item(list(namespace), key=key)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                logger.warning(
                    f"delete_item failed for key={key}, namespace={namespace} (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"delete_item permanently failed for key={key}, namespace={namespace} after {max_retries + 1} attempts: {e}"
                )
                return {"success": False, "namespace":namespace, "key": key}
    return {"namespace": namespace, "key": key, "error": str(last_exc), "success":True}


async def _search_with_retry(
    store,
    namespace: tuple,
    max_retries: int = 2,
    retry_delay: float = 0.5,
):
    """Single delete_item call with retry logic."""
    logger.info(f"search_with_retry: {namespace}")
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            search_results = await store.search_items(list(namespace), limit=1000000)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                logger.warning(
                    f"search_items failed for namespace={namespace} (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"search_items permanently failed for namespace={namespace} after {max_retries + 1} attempts: {e}"
                )
                return {"error": str(last_exc), "namespace":namespace, "success":False}
    return {"search_results": search_results, "error": str(last_exc), "success":True}


async def batch_index_documents_vectorstore(
        client: Any,
        user_id: str, 
        assistant_id: str,
        vectorstore_documents_to_be_indexed: list[any], 
        BATCH_SIZE: int = 1000
    ):

        logger.info(f"BATCH INDEX DOCUMENTS VECTORSTORE BREAKPOINT")


         # Delete documents with the same filename in the metadata
        filenames = [
            doc.metadata.get("filename") 
            for doc in 
            vectorstore_documents_to_be_indexed
            if doc.metadata.get("filename") is not None
        ]

        # Ensure each document has a unique key:
        _ = [
            doc.metadata.update({"document_id":str(uuid.uuid4())})
            for doc in 
            vectorstore_documents_to_be_indexed
            if (doc.metadata.get("document_id") is None) or (type(doc.metadata.get("document_id", []) is not str))
        ]

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
            
        
        # Upload the new documents into the vector store
        logger.info(f"breakpoint before aadd documents")


        num_successful_batch_uploads = 0
        num_successful_batch_searches = 0
        num_successful_batch_deletes = 0
        
        error_batch_documents = []
        error_batch_searches = []
        error_batch_deletes = []

        # batch the document uploads
        if getattr(client.store, "abatch", None) is not None:
            # create upload batch
            batch_put_ops = [PutOp(namespace=(user_id, assistant_id, "document", doc.metadata.get("filename", f"{user_id}'_'{assistant_id}'_document_unknown_filename'")), key=key, value={"page_content":doc.page_content, "metadata":doc.metadata}) for key, doc in zip(insert_document_keys, vectorstore_documents_to_be_indexed)]

            total_documents_to_be_indexed = len(batch_put_ops)

            for i in range(0, total_documents_to_be_indexed, BATCH_SIZE):
            
                batch = batch_put_ops[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (total_documents_to_be_indexed + BATCH_SIZE - 1) // BATCH_SIZE
                try:
                    await client.store.abatch(batch)

                    progress = min(i + BATCH_SIZE, total_documents_to_be_indexed)

                    logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed}")

                    num_successful_batch_uploads += len(batch)

                except Exception as e:
                    logger.info(f"Error in batch {batch_num}: {str(e)}. Attempting Retry.")
                    # Handle the error with a retry
                    try:
                        await client.store.abatch(batch)

                        progress = min(i + BATCH_SIZE, total_documents_to_be_indexed)

                        logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed}")

                    except Exception as e:
                        logger.info(f"Continued error in batch {batch_num}: {str(e)}. Returning batch documents in error")
                        error_batch_documents.extend(batch)
                        continue
        else:
            # asyncio with put_item
            # create upload batch
            batch_put_ops = [((user_id, assistant_id, "document", doc.metadata.get("filename_uuid5", f"{user_id}'_'{assistant_id}'_{str(uuid4())}_document_unknown_filename'")), key, {"page_content":doc.page_content, "metadata":doc.metadata}) for key, doc in zip(insert_document_keys, vectorstore_documents_to_be_indexed)]

            total_documents_to_be_indexed = len(batch_put_ops)

            # delete previous filename chunks
            # extract filename chunk keys
            
            # extract the unique filenames
            filenames_uuid5_set = set([doc.metadata.get("filename_uuid5", "") for doc in vectorstore_documents_to_be_indexed])
            filenames_uuid5_list = list(filenames_uuid5_set)

            batch_search_ops = [(user_id, assistant_id, "document", filename) for filename in filenames_uuid5_list]
            num_search_ops = len(batch_search_ops)



            all_batch_delete_ops = []

            # create all namespace keys for each delete operation through batched searches
            for i in range (0, num_search_ops, BATCH_SIZE):
                batch = batch_search_ops[i : i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (num_search_ops + BATCH_SIZE - 1) // BATCH_SIZE

                progress = min(i + BATCH_SIZE, num_search_ops)

                logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{num_search_ops}; sample namespace: {batch[0]}")

                batch_search_errors = []

                # Search for documents
                search_results = await asyncio.gather(
                    *[_search_with_retry(client.store, namespace=namespace) for namespace in batch]
                )

                # extract the keys
                all_delete_ops = []
                for search_result, namespace in zip (search_results, batch):
                    delete_ops_namespace_keys = [(namespace, item.get("key", "")) for item in search_result.get("search_results", {}).get("items", []) if search_result.get("success", False) is not False]
                    # collect delete operations for the search results
                    all_delete_ops = all_delete_ops + delete_ops_namespace_keys

                # collect delete operations for the batch
                all_batch_delete_ops = all_batch_delete_ops + all_delete_ops

                batch_search_errors = [search_result for search_result in search_results if search_result.get("success", True) is not True]

                batch_search_success = len(batch) - len(batch_search_errors)

                num_successful_batch_searches += batch_search_success
                error_batch_searches.extend(batch_search_errors)

                logger.info(
                    f"BATCH {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed} processed "
                    f"({batch_search_success} success, {len(batch_search_errors)} failed)"
                )                

            num_delete_ops = len(all_batch_delete_ops)

            # Batch delete
            for i in range (0, num_delete_ops, BATCH_SIZE):
                batch = all_batch_delete_ops[i : i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (num_delete_ops + BATCH_SIZE - 1) // BATCH_SIZE

                progress = min(i + BATCH_SIZE, num_delete_ops)

                logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{num_delete_ops}")

                batch_delete_errors = []

                # Search for documents
                delete_results = await asyncio.gather(
                    *[_delete_with_retry(client.store, namespace=batch_delete_op[0], key=batch_delete_op[1]) for batch_delete_op in batch]
                )

                batch_delete_errors = [delete_result for delete_result in delete_results if delete_result.get("success", False) is not False]

                batch_delete_success = len(batch) - len(batch_delete_errors)
                num_successful_batch_deletes += batch_delete_success
                error_batch_deletes.extend(batch_delete_errors)

            # Batch Put ops
            for i in range(0, total_documents_to_be_indexed, BATCH_SIZE):
                batch = batch_put_ops[i : i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (total_documents_to_be_indexed + BATCH_SIZE - 1) // BATCH_SIZE
                progress = min(i + BATCH_SIZE, total_documents_to_be_indexed)

                logger.info(f"Batch {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed}")

                logger.info("breakpoint")

                batch_errors = []

                results = await asyncio.gather(
                    *[_put_with_retry(client.store, namespace=namespace, key=key, value=value) for namespace, key, value in batch]
                )
                
                batch_errors = [result for result in results if result.get("success", True) is not True]
                # batch_errors = [result for result in results if result is not None]
                batch_success = len(batch) - len(batch_errors)

                num_successful_batch_uploads += batch_success
                error_batch_documents.extend(batch_errors)

                logger.info(
                    f"BATCH {batch_num}/{total_batches}: {progress}/{total_documents_to_be_indexed} processed "
                    f"({batch_success} success, {len(batch_errors)} failed)"
                )
        
        if len(error_batch_documents) == 0:
            return {"success": True, "documents_uploaded": num_successful_batch_uploads}
        else:
            return {"success": False, "documents_uploaded": num_successful_batch_uploads, "error_batch_documents": error_batch_documents}

from typing import Any

import logging
logger = logging.getLogger(__name__)


# TEXT TO VECTORSTORE

"""
Text chunking module for processing large text files into Documents
for LangGraph-MongoDB vectorstore with all-MiniLM-L6-v2 embeddings (384 dimensions)
"""

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
        document_id: str,
        filename: str, 
        filename_uuid5: str,
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
                "filename": filename,
                "filename_uuid5": filename_uuid5,
                "document_id":document_id,
            }
        )
        doc.metadata.update(source_metadata)
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
        filename = media_item.get("metadata", {}).get("filename", "")
        filename_uuid5 = str(uuid5(NAMESPACE_URL, filename))

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
   
        token_text_content_length = count_tokens_approximately([text_content])    
        if token_text_content_length > chunk_size and use_semantic_chunks:
        
            # Define meaningful chunks first

            model_structured_output = init_model(
                        configuration=configuration,
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
                length_function=count_tokens_approximately,
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
                        semantic_text_chunk_token_size = count_tokens_approximately([semantic_text_chunk])

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
                                    "filename": filename,
                                    "document_id": str(uuid4()),
                                    "filename_uuid5":filename_uuid5
                                }
                            )
                            doc.metadata.update(source_metadata)
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
                                filename=filename,
                                document_id = str(uuid4()),
                                filename_uuid5 = filename_uuid5
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
                                filename=filename,
                                document_id = str(uuid4()),
                                filename_uuid5 = filename_uuid5
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
                            filename=filename,
                            document_id = str(uuid4()),
                            filename_uuid5 = filename_uuid5
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
                            filename=filename,
                            document_id = str(uuid4()),
                            filename_uuid5 = filename_uuid5
                        )
                if not isinstance(documents, list): # Redundant type verification
                    documents = [documents] 
                all_documents.extend(documents)

            else:
                logger.info(f"use_semantic_chunks is False AND token_text_content_length is less than or equal to the chunk_size (256 tokens before truncation)")
                idx = 0
                text_chunk_token_size = count_tokens_approximately([text_content])
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
                            "filename": filename,
                            "document_id": str(uuid4()),
                            "filename_uuid5":filename_uuid5
                        }
                    )
                    idx += 1
                    doc.metadata.update(source_metadata)

                    if classification_metadata is not None:
                        doc.metadata.update(classification_metadata)
                    docs = [doc]
                    all_documents.extend(docs)

        # Update the total chunks in the metadata:
        result_of_document_metadata_update = [document.metadata.update({"total_chunks": idx}) for document in all_documents]

        return all_documents

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in text chunking during process media item target for vector store: {e}")