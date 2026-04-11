
# Currently have (in process sequence)

## (STEP 1) how i am accepting media (files and urls)
@app.post("/update_avatar_identity_with_media")
async def update_avatar_identity_with_media(
    files: Optional[List[UploadFile]] = File(...),
    url: Optional[str] = None,
    assistant_id: str = None,
    reference_audio: bool = False,
    reference_image: bool = False, 
    proprietary_content: bool = False, 
    current_user: dict = Depends(get_current_user)
):
    # Context user_id, assistant_id
    logger.info(f"UPLOAD MEDIA ENDPOINT ENTRY")
    """
    Upload one or more media files for processing and indexing.
    
    - **files**: One or more files to process
    - **user_id**: User identifier
    - **assistant_id**: Assistant identifier
    """
    try:

        user_id = current_user['identities'][0]['user_id']

        # assitant_config = current_user['app_metadata']['assistant_config']
        # assistant_id = assitant_config['configurable']['assistant_id']  
        # config['configurable'].update(assitant_config['configurable'])
        
        config = {
            "configurable": {
                "user_id": user_id,
                "user_ctx": {"name":None, "description": None},
            }
        }

        config['configurable']['assistant_id'] = assistant_id
        config['configurable']['assistant_ctx'] = {"name":None, "description": None},
        # Read all uploaded files
        media_files = []
        for file in files:
            content = await file.read()
            media_files.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": reference_audio,
                "reference_image": reference_image,               
                "proprietary_content": proprietary_content
            })

# How I am classifying and processing text Example Snippet:


# Process TEXT TO DOCUMENT
async def process_text_to_document(metadata, user_id, assistant_id, media_item) -> list[Document]:
    """ Process text to document; 
    document chunking necessity, 
    situation determination, 
    and future use of the data (vectorstore, analysis, adapter) are handled here   
    """
    logger.info(f"Handling text in process media")
    
    proprietary_content = metadata.get("proprietary_content", False)

    proprietary_content_classification_model = init_model(model_without_tools=True, response_format=ReferenceDocumentOrBiographicalConversationalInformation)
    text_content = media_item.get("content", "")
    
    classification = await proprietary_content_classification_model.ainvoke([SystemMessage(content=REFERENCE_DOCUMENT_OR_BIOGRAPHICAL_CONVERSATIONAL_INFORMATION), HumanMessage(content=text_content[:5000])])
    # if proprietary_content:
    if classification.non_personally_identifiable_information:
        logger.warning(f"proprietary content: No single target; media is only uploaded to vectorstore")
        
        classification_metadata = {
            "classified_situation": "proprietary content",
            "classification_reasoning": classification.reasoning
        }
        
        documents = await process_text_media_item_target_for_vectorstore(
            media_item=media_item, 
            user_id=user_id, 
            assistant_id=assistant_id,
            classification_metadata=classification_metadata,
            use_semantic_chunks=False
        )
        
        for document in documents: 
                    document.metadata.update({"vectorstore_acceptable": True})
        return documents
    
    else:
        logger.info(f"There is a target and the content of the text needs to be analyzed (is this a monologue or multi-speaker or strictly Q & A; how many speakers, etc.)")
        
        # Analyze text situation

        
        model_with_structured_output = init_model(
            model_without_tools=True,
            response_format=TextualSituationalAwareness
        )
        


        system_prompt = TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS
        
        classification = await model_with_structured_output.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_content}
        ])
        logger.info(f"Situation classification: {classification.classified_situation}")

        logger.info(f"Reason for classification : {classification.reasoning}")
        
        classification_metadata = {
            "classified_situation": classification.classified_situation, 
            "classification_reasoning": classification.reasoning
        }
        if classification.classified_situation == "single_speaker":
            # TODO: Determine if the text is of the single person directly speaking as a quote for the entire document in the media item 
            
            # is this content written in first person or is this content about the individual?
            # from src.anubis.utils.prompts.system_prompts import DETERMINE_TEXT_SINGLE_SPEAKER_FIRST_PERSON_TONE_OF_VOICE_SYSTEM_PROMPT

            monologue_vs_distinct_quotes_classification_model = init_model(model_without_tools=True, response_format=MonologuePresentationOrSeriesOfQuotes)

            system_prompt = MONOLOGUE_PRESENTATION_OR_SERIES_OF_QUOTES
            input = [SystemMessage(content=system_prompt), HumanMessage(content=text_content)]

            response = await monologue_vs_distinct_quotes_classification_model.ainvoke(input)
            classification_metadata = { 
                "classified_situation": response.classified_situation,  "classification_reasoning": response.reason
            }
            if response.classified_situation == "SeriesOfDistinctQuotes":
                logger.info("")
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
                            "filename_uuid5":filename_uuid5, 
                            "namespace": "quote"
                        }
                    )
                    idx += 1

                    if classification_metadata is not None:
                        doc.metadata.update(classification_metadata)
                        doc.metadata.update({"vectorstore_acceptable": True})
                    docs = [doc]
                    all_documents.extend(docs)
                [document.metadata.update({"total_chunks": idx}) for document in all_documents]
                
                additional_metadata={
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "created_at": current_timestamp,
                
                            "source": source,
                            "type": "text",
                            "filename": filename,
                            
                            "filename_uuid5":filename_uuid5, 
                            "namespace": "identity",
                            "analysis_acceptable": True
                        }
                analysis_documents = await perform_ocean_analysis(human_message = HumanMessage(content = media_item.get("content")), additional_metadata = additional_metadata)

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
                logger.info(f"proccess_text_media_item_target_for_vectorstore BREAKPOINT in Process media item task: type = text")
                documents = await process_text_media_item_target_for_vectorstore(
                    media_item=media_item, 
                    user_id=user_id, 
                    assistant_id=assistant_id,
                    classification_metadata=classification_metadata,
                    use_semantic_chunks=False
                )
                for document in documents: 
                    document.metadata.update({"vectorstore_acceptable": True})
            # TODO: format for analysis: analyze for content about the target
            # USE THE DETERMINATION TO AUGMENT ANALYSIS AND NOTE THAT THE CONTENT IS EITHER ABOUT THE TARGET OR IS FROM THE TARGET SPEAKING DIRECTLY
        elif classification['classified_situation'] == "q_and_a_dialogue":
            logger.warning(f"Q & A DIALOGUE CLASSIFICATION DETECTED")
            logger.warning(f"""
            # TODO: format for vectorstore; CHUNKS OF NON-TARGET CANNOT BE


## (STEP 6) Create questions from ground truth statements (strings only; will need to process json) src/anubis/utils/dataset/formatting.py

async def create_question_list(str_messages_list: list[str]) -> List[str]:
    class GeneratedQuestionsList(BaseModel):
        question_list: List[str]

    human_messages_list = [HumanMessage(content=message_str) for message_str in str_messages_list]

    model_with_structured_output = init_model(model_without_tools=True, response_format= GeneratedQuestionsList)

    system_message = SystemMessage(content="Given this list of messages, generate a query to which the message is the response. THERE MUST BE A QUESTION FOR RESPONSE AND THE QUESTION ORDER IN THE LIST MUST MATCH THE RESPONSE ORDER. These questions must be succinct.")
    
    messages = [system_message] + human_messages_list

    response = await model_with_structured_output.ainvoke(input=messages)

    return response.question_list


## (STEP 4 & STEP 6) Creating adapter training format from question answer pairs (strings only will need a driver to process json) src/anubis/utils/dataset/formatting.py

""" LLAMA 4 ADAPTER TRAINING FORMAT """

async def llm_single_turn_dataset(question_list: List[str], answer_list: List[str]) -> List[dict]:
    """ Creates a Messages Dataset of Single Turns for a list of question and answer pairs. Used for LLM Adapter Training Format."""
    single_turn_dataset = []
    for question, answer in zip(question_list, answer_list):
        turn = {"messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]}
        single_turn_dataset.append(turn)
    return single_turn_dataset

def llm_multiturn_dataset_one_conversation(question_list: List[str], answer_list: List[str]) -> dict:
    """ Creates a Messages Dataset of a conversation of question and answer pairs. Used for LLM Adapter Training Format.
        This is a single conversation. A list of multiple conversations must be used to for the entire final dataset.
    """
    list_of_messages = []
    for question, answer in zip(question_list, answer_list):
        turn = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        list_of_messages += turn
    multi_turn_dataset = {"messages": list_of_messages}
    return multi_turn_dataset

""" LANGSMITH DATASET FORMAT """

async def langsmith_dataset(question_list: List[str], answer_list: List[str], dataset_source_filename: str) -> List[dict]:
    """ Creates a list of dict example question and answer inputs and outputs """
    examples = []
    examples.append({
        "inputs":{"question": question}, 
        "outputs": {"answer":answer}, 
        "metadata": {"source", dataset_source_filename}} for question, answer in zip(question_list, answer_list))
    return examples



