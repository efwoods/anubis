duplicate learned information as documents learned about the user.


user_id, 'identity')

    # VERIFY FACT DOES NOT ALREADY EXIST
    user_identity_documents_text_list = [document.metadata.get("fact") for document in runtime.state['user_identity_documents']]
    user_content_store_query_results = await runtime.store.asearch(user_identity_namespace, query=user_fact)
    user_content_store_query_results_significant = [item for item in user_content_store_query_results if item.score > 0.8]
    if user_fact in user_identity_documents_text_list or len(user_content_store_query_results_significant) > 0:
        # Fact already exists:

        tool_call_id = runtime.tool_call_id

        update = {"messages": [ToolMessage(content=f"Fact: {user_fact} previously learned", tool_call_id = tool_call_id)]}
        return Command(update=update)
    