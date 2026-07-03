
When there is a typo in the frontend, the memory needs to be edited in two ways:
edit message (message response is edited; the document id is looked up that corresponds to the original message that was sent, the documents are deleted and the new message is re-pipelined throught the graph).

Natural language:

You mispoke: It was ornery as her mother used the "word" on her.... not the "world" on her...

This needs to trigger the same flow as above with the edited message compiled with a model with structured output to edit the original message with the updated message and piecemeal the two together in a logical flow, delete the index, ask for clarification, allow all edits, and re index (or retrieve the original statement, ask for clarification first on the compiled edit, allow the user to accept or manually edit with buttons on the frontend, delete the indexed documents, and re-pipeline through the graph). requires frontend buttons, api approve and edit, langgraph human in the loop interruption, document id metadata, logic to identify the documents that were just created. 


please add an endpoint to list avatar memories (memories may be both in the identity_memory namespace and the memory namespace; must be per user; also include all memories the avatar has about the user who is sending the request; requires assistant_id and user_id) must return a list of the unique memory and the page_content of the memory @src/api/webapp.py; allow update and
  delete endpoints for the memories using the document ID that is associated with the memory; these functions will be used within the logic of the graph at a later time and need to be developed as such please.


# Misinformation correction plan:
<!-- /home/user/.claude/plans/create-a-plan-to-abstract-sun.md -->