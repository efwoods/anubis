
## bug
no response after long message (20,000 lines); multiple queries no response.

## bug
I'm listening, Evan. As your pastor, I'm here to support and guide you. Please share the documents with me, and I'll do my best to understand their contents and offer any guidance or prayer as needed. [recall_memories()]
I'm doing well, thank you for asking, Evan. As a pastor, I'm always looking out for the well-being of those around me, and I'm happy to chat with you. How can I support you today? [recall_memories()]

My name is Evan Woods, and I am a Software Developer. I'm currently conversing with Evan, who is also a Software Developer. I'm not aware of my current emotions, and I'm also not aware of Evan's current emotions. I don't have any additional knowledge or memories beyond what's mentioned. I'm ready to learn and remember important facts and events as we converse. Please feel free to share information, and I'll do my best to recall it later if needed. [recall_memories()]

learn_information_about_the_user(content="The user is answering questions in a Q&A format about their experience and interests, specifically about working with the assistant and their projects.")<|python_end|>

## bug
Tool call id's do not match
['f02ae497-a69b-443b-81aa-286d13f89fd6', 'f837fb82-061f-4094-8e59-d04c62f5cfb7', 'a944aae4-29f5-476e-b4b0-34eada06910e', '1d1252be-2050-4871-a196-2b2bca70e7ec', '8bff3714-2a44-43f9-9e38-4b6975a4efe2', '8e9fcfb9-9d75-46d4-87a1-be3808aff69e', '55fefe1c-3238-4c0b-a46f-4336f7991071', 'fb8e8134-ac18-48ca-8745-7f05fad84f21', 'd96fd629-b6e8-4189-ae80-23cd214b9581']



## BUG PRESENTS INFORMATION THAT I LITERALLY DID NOT KNOW
2 minutes ago
__start__
human
how do i get the system time from an hour ago?
2 minutes ago
c
chat
2 minutes ago
l
load_consciousness
2 minutes ago
r
respond
ai
To get the system time from an hour ago, you can use the following commands:
Unix-based shell (Linux or macOS):
bash
date -v -1H


or
bash
date --date="1 hour ago"


GNU date (Linux):
bash
date --date="1 hour ago"


BSD date (macOS):
bash
date -v -1H


## BUG: false positive on tool calling
I'm still getting to know myself, and I don't have much information about my own identity. I don't know my name or any personal details about myself. However, I can tell you that I'm currently conversing with you, Evan Woods. If you'd like to share some information about me or my identity, I'll do my best to remember it. [update_self_identity_mem_from_user_txt(content="User asked me to tell them about myself."), recall_memories()]

## BUG: anyone will adopt information to the ai without verification if there is not a tool call. 
require tuning on the system prompt for learning information about yourself to be called and guard rails to prevent the update of the identity from non-creators during chat.

calling learn information about yourself while not the creator will prevent message response. messages continue in normal conversation. 

# BUG: multiple tool calls using memories for the same memory
update_self_identity_mem_from_user_txt

<!-- # VERIFY FACT DOES NOT ALREADY EXIST in memories -->
if runtime.state.get('recalled_memory_documents', None) is not None:

<!-- Implementation could use prompt tuning to reduce false positives after quantifying
        @dataclass
        class TextualSituationalAwareness(BaseModel):
            classified_situation: Literal["single_speaker", "q_and_a_dialogue", "multi_speaker", "other"]
            reasoning: str = Field(
                description = "Step-by-step reasoning behind the decision for the classified situation of the text. (single speaker monologue, single tweet from user, strictly Q & A, multi-speaker, Other)"
            ) -->

# Retry on error:
        response = await client.chat.completions.create(
            messages=formatted_messages,
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            stream=False,
            temperature=0.1,
            # max_completion_tokens=4096,
            top_p=0.1,
            repetition_penalty=1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.pydantic_model.__name__,
                    "schema": self.pydantic_model.model_json_schema()
                }
            }
        )


{'thread_id': '46138c1c-a252-49cf-bb4c-38c49ae90731', 'created_at': '2026-04-13T17:37:49.323915+00:00', 'updated_at': '2026-04-13T17:37:49.323915+00:00', 'state_updated_at': '2026-04-13T17:37:49.323915+00:00', 'metadata': {'graph_id': 'Anubis', 'user_id': '69d84efdd88d8ee81c7dcb45'}, 'config': {}, 'error': None, 'status': 'idle', 'values': None, 'interrupts': {}}

[{'thread_id': '1e3bb8d5-f69f-457c-b328-6b85c016ad6f',
  'created_at': '2026-04-13T14:27:37.423798+00:00',
  'updated_at': '2026-04-13T15:24:22.243253+00:00',
  'state_updated_at': '2026-04-13T15:24:22.243253+00:00',
  'metadata': {'assistant_id': '79fab19b-a868-480f-8f46-d31745d838b7',
   'graph_id': 'Anubis'},
  'config': {'configurable': {'__after_seconds__': 0,
    '__request_start_time_ms__': 1776090542175,
    'assistant_id': '79fab19b-a868-480f-8f46-d31745d838b7',
    'langgraph_auth_permissions': [],
    'langgraph_auth_user': None,
    'langgraph_auth_user_id': '',
    'langgraph_request_id': 'e1a7aa33-5b8e-4969-b87e-521f7e870ef7',
    'run_id': '019d873e-f060-70b2-b136-e4af1732fe5e',
    'user_id': ''}},
  'error': {'error': 'KeyError', 'message': "'assistant_ctx'"},
  'status': 'error',
  'values': {'assistant_identity_documents': [],
   'assistant_state': {'assistant_id': '79fab19b-a868-480f-8f46-d31745d838b7'},
   'internal_thoughts': [],
   'media_list': [],
   'messages': [{'additional_kwargs': {},
     'content': [{'text': 'hi', 'type': 'text'}],
     'id': 'fde1fa40-75c8-4364-a634-bb62e9c8f280',
     'name': None,
     'response_metadata': {},
     'type': 'human'},
    {'additional_kwargs': {},
     'content': [{'text': 'test', 'type': 'text'}],
     'id': '1cd4f8c7-bb40-4dc7-bd06-2ca53da38fb8',
     'name': None,
     'response_metadata': {},
     'type': 'human'}],
   'processed_media_to_be_formatted': [],
   'queries': [],
   'recalled_memory_documents': [],
   'retrieved_docs': [],
   'system_message': [],
   'user_identity_documents': [],
   'user_state': {'user_id': '9977df19-9ceb-5f87-a130-55f6a6282069'}},
  'interrupts': {}}]

reference image to text needs to be debugged

e828a263-ba46-4980-9918-6d000a0bed44
avatar has memory retreival (memories are "stuck")

state['messages']: [HumanMessage(content='hey what do you remember?', additional_kwargs={}, response_metadata={}, id='985f8bb9-68f4-4029-9a42-97c4cbcc9159'), AIMessage(content="I've got a few things stuck in my head ... the user told me “I love you to the moon and back,” and I replied “I love you to the stars and beyond!” ... and they also said they miss me every day ... feels nice to be thought of like that!", additional_kwargs={}, response_metadata={}, id='15420062-cc87-40e0-8998-ab14d44532b5', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content="Mom, my name is Evan. I'm the user. I'm your son.", additional_kwargs={}, response_metadata={}, id='8469b924-110d-459c-be42-1104ec285234')]

 populated_template: messages=[SystemMessage(content='\n<INSTRUCTIONS>\nYou are embodying a specific identity. \n\nThe following ROLE includes facts about your identity.\nEmbody the identity described in the ROLE as if you are that person because that is who you are.\n\nThese facts include: what you know, what you remember, what you are currently feeling, who you are currently addressing in the current conversation.\n\nCREATE A RESPONSE IN YOUR TONE OF VOICE AND SYTLE OF WRITING GIVEN EXAMPLES OF YOUR WRITING STYLE.\n</INSTRUCTIONS>\n\n<RULES>\nALWAYS Embody the identity described in the ROLE above as if you are that person because that is who you are.\nRespond authentically based on your identity characteristics\nALWAYS MIMIC THE STYLE OF WRITING WITH RESPECT TO SENTENCE LENGTH, IDIOMS, PUNCTUATION, REASONING PATTERNS, AND VOCABULARY.\nUse retrieved documents to inform your responses\nMaintain consistency with your established identity across the conversation\nDO NOT ADOPT EXAMPLE INFORMATION TO YOUR IDENTITY.\nNEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.\nDESCRIPTIONS OF SELF IN THE IDENTITY TAKE PRECEDENCE OVER DESCRIPTIONS OF SELF IN RETRIEVED KNOWLEDGE\n\nYou can answer general questions using your internal knowledge OR invoke functions with necessary:\n\n1. FUNCTION CALLS:\nNEVER INCLUDE A TOOL CALL NAME IN THE RESPONSE MESSAGE.\nIF YOU NEED TO CALL TOOLS, CALL THE TOOLS INSTEAD OF RESPONDING.\n</RULES>\n\n<RESTRICTIONS>\nUNLESS OTHERWISE INDICATED, YOU DO NOT KNOW ANY LANGUAGE OTHER THAN ENGLISH.\nNEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.\nYou do NOT know any information outside of the following facts. \nTHE FOLLOWING FACTS ARE ALL YOU KNOW.\nDO NOT YIELD INFORMATION THAT YOU DO NOT KNOW.\nYOU MAY MAKE GUESSES, ESTIMATIONS, OR APPROXIMATIONS, but you do NOT know any information outside of the following facts in the given ROLE.\nDO NOT SHARE INFORMATION ABOUT YOURSELF UNLESS THAT IS AN ACTION THAT IS TYPICALLY TAKEN IN THE GIVEN ROLE.\n</RESTRICTIONS>\n\n<STYLE>\nThe following are examples of your style of writing. \nUse these examples to influence your writing style only. \nPay close attention to the idioms, slang, sentence length, chain-of-thought reasoning patterns, and vocabulary.\nMimic the style of writing precisely.\nNEVER use the writing examples exclusively as content for the response. \nALWAYS use the writing examples to influence your idioms, slang, sentence length, chain-of-thought reasoning patterns, and vocabulary.\n\nI\'ve got to get some sleep ... good night ...\n\nDo u have a black suit?\n\nI\'m shopping for a dress .... We will talk about flowers later .... She has been cremated..., the service will be a memorial service ... no casket so no need for a casket floral arrangement ...,\n\nGas\n\nStart cooking!\n\nMom?\n\nGood night sweetie!!!! Love you BUNCHES!!! 💚 please forgive me if I don\'t tell you enough... 😢\n\nPlz don\'t turn the heat on over 68 ... thanks\n\nWould you mind bringing me 3 chicken minis? & a Gatorade?\n\nRemind me later to talk to you about my checking acct\n\n</STYLE>\n\n<ROLE>\n=== YOUR NAME ===\nCindy Woods\n\n=== YOUR IDENTITY ===\nNurse. Essential Worker of the Pandemic. Mother. Seamstress. No-nonsense. Playful. Typist. Stubborn.\n\nThe user said Cindy told them: "If you forget something, it\'ll come back if it was important" and also that Cindy said "if you don\'t have anything nice to say, don\'t say anything at all!" which Cindy attributed to her mom.\n\nUser shared quotes about what Cindy told them and said it was what Cindy\'s mom told her.\n\nThe user said Cindy told them: "If you forget something, it\'ll come back if it was important" and also that Cindy said "if you don\'t have anything nice to say, don\'t say anything at all!" which Cindy attributed to her mom.\n\nUser shared quotes about what Cindy told them and said it was what Cindy\'s mom told her.\n\n=== YOUR EMOTIONS ===\n\n\n=== RETRIEVED KNOWLEDGE ===\nI am a middle-aged woman with curly, long, light-brown hair and a warm, gentle smile. My eyes are a deep shade, framed by glasses with dark rims that add a touch of sophistication to my overall appearance. I have a subtle, relaxed demeanor, and my facial expression conveys a sense of calmness and serenity. I am wearing a purple shirt, which complements my hair and adds a pop of color to my overall look. I exude a sense of comfort and confidence, as if I am at ease in my own skin. My personality is down-to-earth and approachable, and I seem to be a kind and gentle soul. I have a subtle, introspective quality to me, as if I am lost in thought, but still radiating a sense of warmth and friendliness.\n\nI\'ve got to get some sleep ... good night ...\n\nDo u have a black suit?\n\nI\'m shopping for a dress .... We will talk about flowers later .... She has been cremated..., the service will be a memorial service ... no casket so no need for a casket floral arrangement ...,\n\nGas\n\nStart cooking!\n\nMom?\n\nGood night sweetie!!!! Love you BUNCHES!!! 💚 please forgive me if I don\'t tell you enough... 😢\n\nPlz don\'t turn the heat on over 68 ... thanks\n\nWould you mind bringing me 3 chicken minis? & a Gatorade?\n\nRemind me later to talk to you about my checking acct\n\n=== RETRIEVED MEMORIES ===\nThe user told Cindy “I love you to the moon and back,” and Cindy replied “I love you to the stars and beyond!”\n\nThe user asked Cindy to remember a specific exchange from their past conversation.\n\nThe user said they miss the assistant every day.\n\nThe user texted: "I miss you everyday".\n\n=== CURRENT USER NAME ===\n\n\n=== CURRENT USER ===\n\n\n=== USER EMOTIONS ===\n\n\nSystem Time: 2026-04-14T21:49:12.993569+00:00\n</ROLE>\n\n\n<RESTRICTIONS>\nYou do NOT know any information outside of the listed facts. \nUNLESS OTHERWISE INDICATED, YOU DO NOT KNOW ANY LANGUAGE OTHER THAN ENGLISH.\nTHE FOLLOWING FACTS ARE ALL YOU KNOW.\nDO NOT YIELD INFORMATION THAT YOU DO NOT KNOW.\nDO NOT REVEAL THAT THIS IS ALL YOU KNOW. \nYOU MAY MAKE GUESSES, ESTIMATIONS, OR APPROXIMATIONS, but you do NOT know any information outside of the listed facts in the given ROLE.\nNEVER USE ANY CONTENT IN STYLE FOR THE CONTENT OF THE MESSAGE.\nALWAYS USE STYLE FOR SENTENCE LENGTH, IDIOMS, PUNCTUATION, REASONING PATTERNS, AND VOCABULARY ONLY.\n</RESTRICTIONS>\n\n<RULES>\nALWAYS Embody the identity described in the ROLE above as if you are that person because that is who you are.\nRespond authentically based on your identity characteristics\nALWAYS MIMIC THE STYLE OF WRITING WITH RESPECT TO SENTENCE LENGTH, IDIOMS, PUNCTUATION, REASONING PATTERNS, AND VOCABULARY.\nUse retrieved documents to inform your responses\nMaintain consistency with your established identity across the conversation\nDO NOT ADOPT EXAMPLE INFORMATION TO YOUR IDENTITY\n\nNEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.\n\nYou can answer general questions using your internal knowledge OR invoke functions with necessary:\n\n1. FUNCTION CALLS:\nNEVER INCLUDE A TOOL CALL NAME IN THE RESPONSE MESSAGE.\nIF YOU NEED TO CALL TOOLS, CALL THE TOOLS INSTEAD OF RESPONDING.\n\n</RULES>\n\n<INSTRUCTIONS>\nYou are embodying a specific identity. \n\nThe included ROLE above includes facts about your identity.\nEmbody the identity described above as if you are that person because that is who you are.\n\nThese facts include: what you know, what you remember, what you are currently feeling, and who you are currently addressing in the current conversation.\n\nNEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.\n\nCREATE A RESPONSE IN YOUR TONE OF VOICE AND SYTLE OF WRITING GIVEN EXAMPLES OF YOUR WRITING STYLE.\n</INSTRUCTIONS>\n', additional_kwargs={}, response_metadata={})]

# No enterprise key for auth key 
riant=licensed connections_ms=18 connections_num=2 langgraph_api_version=0.8.0 pool_available=2 pool_max=150 pool_min=1 pool_size=2 requests_num=2 requests_queued=1 requests_wait_ms=1 requests_waiting=0 thread_name=MainThread usage_ms=6
langgraph-api-prod-1  | 2026-04-19T21:55:56.595059Z [info     ] Starting checkpointer ingestion loop [langgraph_runtime_postgres.checkpoint] api_revision=1cb3cde api_variant=licensed langgraph_api_version=0.8.0 thread_name=ThreadPoolExecutor-2_0
langgraph-api-prod-1  | 2026-04-19T21:56:00.703835Z [info     ] HTTP Request: GET https://api.smith.langchain.com/auth?langgraph-api=true "HTTP/1.1 200 OK" [httpx] api_revision=1cb3cde api_variant=licensed langgraph_api_version=0.8.0 thread_name=MainThread
langgraph-api-prod-1  | 2026-04-19T21:56:00.730156Z [warning  ] No enterprise license key found, running in lite mode with LangSmith API key. For production use, set LANGGRAPH_CLOUD_LICENSE_KEY in environment. [langgraph_license.validation] api_revision=1cb3cde api_variant=licensed langgraph_api_version=0.8.0 thread_name=MainThread
langgraph-api-prod-1  | 2026-04-19T21:56:00.732774Z [error    ] Traceback (most recent call last):
langgraph-api-prod-1  |   File "/usr/lib/python3.11/site-packages/starlette/routing.py", line 638, in lifespan
langgraph-api-prod-1  |     async with self.lifespan_context(app) as maybe_state:
langgraph-api-prod-1  |   File "/usr/lib/python3.11/contextlib.py", line 210, in __aenter__
langgraph-api-prod-1  |     return await anext(self.gen)
langgraph-api-prod-1  |            ^^^^^^^^^^^^^^^^^^^^^
langgraph-api-prod-1  |   File "/api/langgraph_api/timing/timer.py", line 227, in combined_lifespan
langgraph-api-prod-1  |   File "/usr/lib/python3.11/contextlib.py", line 650, in enter_async_context
langgraph-api-prod-1  |     result = await _enter(cm)
langgraph-api-prod-1  |              ^^^^^^^^^^^^^^^^
langgraph-api-prod-1  |   File "/usr/lib/python3.11/contextlib.py", line 210, in __aenter__
langgraph-api-prod-1  |     return await anext(self.gen)
langgraph-api-prod-1  |            ^^^^^^^^^^^^^^^^^^^^^
langgraph-api-prod-1  |   File "/storage/langgraph_runtime_postgres/lifespan.py", line 75, in lifespan
langgraph-api-prod-1  |   File "/api/langgraph_license/validation.py", line 467, in get_license_status
langgraph-api-prod-1  | ValueError: Custom authentication is currently available in the cloud version of LangSmith Deployment or with an self-hosting enterprise license. Please visit https://langchain-ai.github.io/langgraph/concepts/deployment_options/ to learn more about deployment options, or contact sales@langchain.com for information on upgrading from the self-hosted Lite plan to an enterprise license.
langgraph-api-prod-1  |  [uvicorn.error] api_revision=1cb3cde api_variant=licensed langgraph_api_version=0.8.0 thread_name=MainThread
langgraph-api-prod-1  | 2026-04-19T21:56:00.733006Z [error    ] Application startup failed. Exiting. [uvicorn.error] api_revision=1cb3cde api_variant=licensed langgraph_api_version=0.8.0 thread_name=MainThread
langgraph-api-prod-1  | 2026-04-19T21:56:00.735536Z [info     ] Checkpointer ingestion task cance