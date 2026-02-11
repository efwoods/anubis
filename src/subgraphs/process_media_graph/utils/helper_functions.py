from typing import Any

import base64

import logging
logger = logging.getLogger(__name__)

# AUDIO PROCESSING

from functools import lru_cache
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

@lru_cache(maxsize=1)
def get_whisper_pipeline():
    """Load and cache the Whisper model pipeline"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe


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
        idx += 1
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
        if classification_metadata is not None:
            doc.metadata.update(classification_metadata)
        documents.append(doc)
    
    return idx, documents

async def process_text_media_item_target_for_vectorstore(
    media_item: Dict[str, Any],
    user_id: str,
    assistant_id: str,
    configuration: GlobalConfiguration,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
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
    
    if len(text_content) >= chunk_size and use_semantic_chunks:
    
        # Define meaningful chunks first
    
        tools = []
    
        model = init_model(
                    configuration.provider_model,
                    configuration.llama_api_base_url,
                    configuration.llama_api_key,
                    tools,
                    configuration.dev,
        )
        
        SEMANTICALLY_CHUNK_TEXT_SYSTEM_PROMPT = """ <Instructions> 
        - Do not alter the text in any way. 
        - Keep all text as originally sent. 
        - When there are reasonable sections to split the text. 
        - Return a list of the text portions. 
        - Attempt to create lists with the number of characters less than {chunk_size} characters if possible otherwise maintain the semantically meaningful chunk.
        - The length of the number of characters in each chunk takes secondary precedence to chunks that are semantically meaningful and is an OPTIONAL requirement.
        </Instructions> 
        
        <Example>

        For Example, The following text would be separated into three chunks, one for each given example labeled First Example, Second Example, and Third Example:
        
        # Neuralink_application

        ## First Example
        I worked with a team of engineering students to build a robot to sort washers based on color and weight. 
        
        Specifically, the goal of the Clemson University ECE 4950 Senior Design project was to create a system which used an electromagnet to move a washer to a load cell where it could be weighed, and use this measurement to sort the washer into the correct slot on a circular disk. 
        
        The project required the construction of an instrumentation amplifier for the amplification of a signal output from a load cell. 
        
        The load cell implemented a strain gauge in a Wheatstone bridge to prove the output. Before amplification, the measured signal from the load cell changed only by a fraction of a millivolt when loaded. The electromagnet was powered by a Techron 7541 Power Amplifier because the lab's Quanser Q4 board could not supply enough current to power the electromagnet. 
        
        It became possible to interface between the sensor, actuator, and the host PC with precision once I applied an algorithm to rotate the disk containing the washers with incremental degrees for n rotations. Prior to this, we would send the entire degree of rotation to the Arduino Uno microcontroller, and the board would rotate too far or too short. 
        
        By signaling the board to rotate in degrees that were an order of magnitude less than prior, we could precisely place, weigh, and sort washers by both weight and color safer and faster than any other team. 
        
        The project resulted in a reliable and efficient system which could distinguish between the weights of an arbitrary number of washers and in response sort them into a desired arrangement which was shown by a MatLab GUI at runtime. 
        
        I narrated a short video to showcase the project which is viewable here: https://youtu.be/f9yRaDn7Hp8
        
        ## Second Example
        
        Through the MIT Applied Data Science Program, I worked on a project which used deep learning to create a machine learning model that accurately classifies the emotion of an input image of an individual's face. 
        
        The four detected emotions were happy, sad, surprised, & neutral. 
        
        I implemented techniques such as data augmentation, hyperparameter tuning, early stopping callbacks and transfer learning to more than double prior accuracies of the base model of EfficientNet v0.2 and ultimately reach 60% accuracy in 10 epochs in 80 minutes on a CPU for no-cost. 
        
        This design was efficient as well as cost effective. 
        
        ## Third Example
        
        Using AWS Sagemaker, I created a machine-learning-based solution to assist in shortlisting candidates with a higher chance of VISA approval to expidite and simulate the Office of Foreign Labor Certification's process of processing applications for different positions for temporary or permanent labor certifications. 
        
        This entailed administering jobs through AWS Sagemaker to process, split, clean, & store training & testing data in an S3 Bucket. 
        
        I further created training jobs to train a model using a decision tree classification method. 
        
        The performance of the model was appraised using a confusion matrix to compute the accuracy, precision, & recall scores based on the true negative, false positive, false negative, and true positive predictions of the model. 
        The hyperparameters of the model were further tuned to reach a Final Objective Value of 0.787 before the artifact was saved and ultimately deployed to queryable endpoint. 

        </Example>

        <Rules>
        - Do not change any of the original text.
        - Return all text that was originally sent.
        - If there are no meaningful sections or portions, then DO NOT separate the text and only return a list containing a single item of the original text.
        - Always return a list.
        - Attempt to create lists with the number of characters less than {chunk_size} characters if possible otherwise maintain the semantically meaningful chunk.
        - The length of the number of characters in each chunk takes secondary precedence to chunks that are semantically meaningful and is an OPTIONAL requirement.
        </Rules>

        """

        formatted_system_prompt = SEMANTICALLY_CHUNK_TEXT_SYSTEM_PROMPT.format(chunk_size=chunk_size)

        model_result = await model.ainvoke(input=[
            {"role": "system_prompt", "content": formatted_system_prompt}, 
            {"role": "user", "content": text_content}
        ])

        logger.info(f"model_result breakpoint on semantic chunking of text: {model_result}")

        # Initialize recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

        if isinstance(model_result, list):
            # Ensure the model has not altered the text in any way (length of text is the same before and after invocation)
            text_total_text_length_of_semantic_chunks = 0 
        
            for chunk in model_result:
                text_total_text_length_of_semantic_chunks += len(chunk)
        
            if text_total_text_length_of_semantic_chunks == text_content.__len__():
                # Create Document objects for each chunk
                all_documents = []
                current_timestamp = datetime.now(tz=timezone.utc).isoformat()
                idx = 0

                for semantic_text_chunk in model_result:

                    if len(semantic_text_chunk) < chunk_size:
                        """ create documents of semantic chunks """
                        idx += 1
                        
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
            all_documents.extend(documents)
    else:
        if len(text_content) >= chunk_size:
            logger.info(f"length of text context is not greater than the chunk size")
        else:
            logger.info(f"use_semantic_chunks is False")
        
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
        all_documents.extend(documents)

    # Update the total chunks in the metadata:
    updated_documents = [document.metadata.update({"total_chunks": idx+1}) for document in all_documents]
    return updated_documents
            
    
