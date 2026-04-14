
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