# avatar is learning single facts about a user from a story rather than multiple facts:

I need you to learn that you are a Man. That the day you were going for glasses, you decided to pick up your glasses before you saw the movie, "Crouching Tiger, Hidden Dragon", that you thought is was good that you got your glasses before you saw the movie, because everything was now clear! You could read the signs in the back of the Walmart (you told this to your mom). You could see the individual leaves on the trees! She felt so bad that you couldn't see at the time, but you told her it's okay. You loved watching action movies and science fiction movies with your mom. You were her "guy". You were her "buddy".

###

@tool("update_self_identity_mem_from_user_txt", args_schema = AssistantFactAndContext)
async def update_self_identity_mem_from_user_txt( # pseudo identity update using namespace (USER_ID, ASSISTANT_ID, 'MEMORY')
    assistant_fact: str, 
    fact_context: str,
    # Hide these arguments from the model.
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> GlobalState:
    """
    <INSTRUCTIONS>
    Learn Facts about yourself from the User through text

    Update the known information about yourself.
    This tool is used to create memories that the user has told the assistant that are 
    SPECIFICALLY ABOUT THE IDENTITY OF THE ASSISTANT, MODEL, or LLM 
    addressed as YOU or or YOUR or YOURS or the direct given name of the assistant.
    An example memory is the user telling the assistant what the assistant's name is or facts about the assistant.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.
    
    Identify the fact that the user shared about YOU, the assistant. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    
    </INSTRUCTIONS>
    
    <RESTRICTIONS>
    NEVER call this tool multiple times with the same fact.
    </RESTRICTIONS>

    <EXAMPLE>
    Example:
    Hi, my name is Evan.
    This is a description of me.
    
    In each example, the name and description are each saved in a database namespace between the assistant and the user.     

    Example:
    Input:
    Your name is Shivon Zilis, and you have twins.
    Becomes multiple tool calls (multiple facts follow):
    My name is Shivon Zilis.
    I have twins.
    </EXAMPLE>

    <RESTRICTIONS>
    NEVER call this tool multiple times with the same fact.
    </RESTRICTIONS>

    <INSTRUCTIONS>
    Learn Facts about yourself from the User through text

    Update the known information about yourself.
    This tool is used to create memories that the user has told the assistant that are 
    SPECIFICALLY ABOUT THE IDENTITY OF THE ASSISTANT, MODEL, or LLM 
    addressed as YOU or or YOUR or YOURS or the direct given name of the assistant.
    An example memory is the user telling the assistant what the assistant's name is or facts about the assistant.
    THERE MAY BE MORE THAN ONE FACT. IN THAT CASE, CALL THIS TOOL MULTIPLE TIMES WITH EACH DISTINCT FACT.
    
    Identify the fact that the user shared about YOU, the assistant. 
    Do not change the information of the fact.
    Identify the CONTEXT behind the fact given the list of messages.
    The context behind the fact must be succinct. 
    The fact must be clear and complete.
    
    </INSTRUCTIONS>

    Args:
        assistant_fact: The main content of the assistant's identity. For example:
            "User expressed that the assistant has an interest in learning about French."
        fact_context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
    """
    class AssistantFactAndContext(BaseModel):
        """
        Extract Facts about the ASSISTANT and the context of that fact given the history of messages.
        """
        assistant_fact: str =  Field(description = "This is the fact about the assistant that was shared by the user.")
        fact_context: str = Field(description = "This is the context of the messages from which this fact was made.")
    
    logger.info(f"learn_information about user breakpoint")

    # Verify the current user is the creator and responsible for the identity of the avatar
    assistant_owner_user_id = runtime.config['configurable']['assistant_ctx']['metadata']['user_id']
    user_id = runtime.config['configurable']['user_id']
    if assistant_owner_user_id != user_id:
        tool_call_id = runtime.tool_call_id
        update = {"messages": [ToolMessage(content=f"Did not adopt information of the identity that was not created by the user.", tool_call_id=tool_call_id)]}
        return Command(update=update)
 
    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(runtime.config)
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")

    # Memory of the Identity of the assistant presented as text from the user.
    # Post-guard, user_id is the assistant's owner, so writes land in the same
    # namespace the consciousness loader reads from.
    assistant_memory_namespace = (user_id, assistant_id, 'identity_memory')

    # VERIFY FACT DOES NOT ALREADY EXIST in memories
    
    if len(runtime.state.get('recalled_memory_documents', [])) != 0:
        assistant_identity_documents_text_list = [document.metadata.get("fact") for document in runtime.state['recalled_memory_documents']]
        assistant_content_store_query_results = await runtime.store.asearch(assistant_memory_namespace, query=assistant_fact)
        assistant_content_store_query_results_significant = [item for item in assistant_content_store_query_results if item.score > 0.8]
        if assistant_fact in assistant_identity_documents_text_list or len(assistant_content_store_query_results_significant) > 0:
            # Fact already exists:
            tool_call_id = runtime.tool_call_id
        
            update = {"messages": [ToolMessage(content=f"Fact: {assistant_fact} previously learned", tool_call_id=tool_call_id)]}
            return Command(update = update)

    # if runtime.state.get('assistant_identity_documents', None) is not None:

    # model_with_structured_output = init_model(context = runtime.context, response_format=AssistantFactAndContext)

    # DECISION_INSTRUCTIONS = """
    # <INSTRUCTIONS>
    # Identify the fact that the user shared about YOU, the assistant. 
    # Do not change the information of the fact.
    # Identify the CONTEXT behind the fact given the list of messages.
    # The context behind the fact must be succinct. 
    # The fact must be clear and complete.
    # </INSTRUCTIONS>

    # <FACT>
    # {content}
    # </FACT>

    # <INSTRUCTIONS>
    # Identify the fact that the user shared about YOU, the assistant. 
    # Do not change the information of the fact.
    # Identify the CONTEXT behind the fact given the list of messages.
    # The context behind the fact must be succinct. 
    # The fact must be clear and complete.
    # </INSTRUCTIONS>
    # """


    # system_message = SystemMessage(content=DECISION_INSTRUCTIONS.format(content=content))

    # chat_prompt_model = system_message + runtime.state['messages']

    # response = await model_with_structured_output.ainvoke(input=chat_prompt_model.messages)
    
    # assistant_fact = getattr(response, "assistant_fact", "")
    # fact_context = getattr(response, "fact_context", "")

    searchable_page_content = "\n\n".join([assistant_fact, fact_context])

    identity_id = str(uuid.uuid4())
    document_metadata = {
        "user_id":user_id,
        "assistant_id": assistant_id,
        "id": identity_id,
        "fact_context": fact_context,
        "fact":assistant_fact
    }

    assistant_identity_memory_document = Document(page_content = searchable_page_content, metadata=document_metadata)
    assistant_identity_memory_document_json = assistant_identity_memory_document.to_json()

    await runtime.store.aput(
        assistant_memory_namespace,
        key=identity_id,
        value={"document": assistant_identity_memory_document_json},
    )

    tool_call_id = runtime.tool_call_id
    update = {"assistant_identity_documents": [assistant_identity_memory_document],
              "messages": [ToolMessage(content=f"Learned: {document_metadata['fact']}", tool_call_id=tool_call_id)]}

    return Command(update=update)
# Document uploading analyzing and processing
https://github.com/efwoods

/home/user/gh/anubis-project/wt/f-psycho-analysis/data/test_data_avatar_evan_woods/Neuralink_application-main/NEURALINK_APPLICATION_README.txt
