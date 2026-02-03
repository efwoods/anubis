I need to build a graph to allow for the CRUD of assistants, 
short-term memory, 
long-term memory, 
list of conversations (each conversation is a thread), 
document uploading to a vectorstore, 
and retrieval augmented generation based on conditional tool use. 

a particular user with a user_id creates an assistant. 

I need to list the assistants for the user, 
CRUD the assistants, 
have all the assistants have names and information of the assistant identity that is prompt-**injectible** for self awareness and does not change between conversations.  

The assistant that is created needs to have memory of relationships with the user and previous conversations. 

I need short and long term memory with conversations, CRUD of assistants that are filterable by user, persistent assistant schema that is defineable to the assistant and prompt injectible for self awareness, and chat inference with media (9 images), retrieval-augmented generation, and memory creation/storage/retrieval tonight. 

Conversations need to be maintained as threads per user_id per avatar_id (assistant) and media uploaded through chat needs to be seen every time the user revisits a conversation. 

The user can ask questions about the media and the assistant responds. 
The assistant remembers facts about the user when the user tells the assistant who the user is. 
The assistant knows facts about itself. 


# Today
1. Design and implement a graph structure for CRUD operations on assistants, short-term memory, long-term memory, conversation threads, document uploads to vectorstore, and retrieval-augmented generation with conditional tool use.

2. Enable user_id-based creation of assistants.

3. Implement listing and CRUD for assistants, filtered by user_id.

4. Define assistant schema with persistent names and identity info for prompt-injection and self-awareness, unchanged across conversations.

5. Integrate short-term and long-term memory for assistants to retain user relationships and previous conversations.

6. Develop memory creation, storage, and retrieval mechanisms tied to conversations.

7. Create conversation threads per user_id and avatar_id (assistant).

8. Support chat inference with up to 9 images, RAG, and memory handling.

9. Ensure uploaded media persists and is visible in revisited conversations; enable user queries about media with assistant responses.

10. Implement assistant memory for user facts shared during interactions.

11. Enable assistant self-knowledge of its own facts.

############
VECTORSTORE ADD AND RETRIEVE
CONVERT MEDIA
PROMPT-DOCUMENT ANALYSIS OF TEXT WITH LLM (TOOL)


### 
prompt for memories and add memories to the identity in conversation ; this is a recollection with a source as "recollection from user_id"

Q and Answer format for turn-style recollection and memory addition

CHAT WITHOUT DOUBLE MESSAGES
TOOL USE HEALTH CHECK
UPDATE IDENTITY OF ASSISTANT AND USER
STORE MEMORIES IN STORAGE
INDEX DOCUMENTS INTO VECTORSTORE
RETRIEVE DOCUMENTS FROM VECTORSTORE
