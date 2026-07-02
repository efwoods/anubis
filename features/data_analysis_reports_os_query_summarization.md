I want to replace the model in think with a deep agent capable of implementing summarization middleware, creating analysis and reports on data, and querying the OS on data without compromising
  on any of the current capabilities;

  @conversation_summarization.md (1-2) 
@src/anubis/graph.py @src/anubis/utils/nodes.py 

I need to summarize the conversation if the context is too long.


 @src/anubis/utils/model.py @features/prompt_drafts/_4deep_agent_for_analysis_reports_os_query_summarization.md I am preparing to extend agent capabilities. I want to use a deep agent rather than a model for the avatar model with tools during think:  
async def think(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""

    """ CREATE MODEL """

    # model invocation
    # TODO: response_metrics_aggregation
    avatar_model_with_tools = init_model(
        context=runtime.context,
        tools=identity_tools,
    )

    # logger.info(f"breakpoint")
    messages = state["system_message"] + state["messages"] + state["internal_thoughts"]

    # TODO: response_metrics_aggregation
    writer = get_stream_writer()
    merged: AIMessageChunk | AIMessage | None = None
    async for chunk in avatar_model_with_tools.astream(messages):
        merged = chunk if merged is None else merged + chunk
        # Stream only while the running merge has no tool calls, so tool turns do not
        # emit assistant_token events meant for the final user-visible reply.
        if not (getattr(merged, "tool_calls", None) or []):
            delta = chunk.content
            if isinstance(delta, str) and delta:
                writer({"type": "assistant_token", "text": delta})
    assert merged is not None
    response = _coalesce_ai_message(merged)
    avatar_response_content = getattr(response, "content")
    logger.info(f"Avatar Model Response: {avatar_response_content}")
    if response.tool_calls:
        return {"internal_thoughts": [response]}
    _attach_go_emotions_metadata(response)
    return {"internal_thoughts": [response], "messages": [response]}

 
the model is initialized here: @src/anubis/utils/model.py  

It is critical that the current existing logic remains. I need to use deep agents with summarization middleware to start and, in the future, (do not implement these yet; keep them in mind however) extend the capabilities when appropriate (creating
  slides and reports for presentations; using a search engine. I should be able to continue to use the graph functionality completely (load conscioussness, think, process thoughts, optionally load consciousness again... respond), as is while also using deep
  agents with summarization middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization


# Data analysis agent:
integrate api with a slack bot to accept messages and files and perform all features such as email creation and sending, and querying personal information if the user is logged in. 
https://docs.langchain.com/oss/python/deepagents/data-analysis
https://www.google.com/search?client=ubuntu-sn&channel=fs&q=slack+messaging+chatbot+llm
https://docs.slack.dev/tools/bolt-python/getting-started/
https://docs.slack.dev/tools/bolt-python/concepts/web-api/
https://developer.nvidia.com/blog/create-a-custom-slackbot-llm-agent-with-nvidia-nim-and-langchain/


https://docs.langchain.com/oss/python/deepagents/code/overview#deep-agents-code


