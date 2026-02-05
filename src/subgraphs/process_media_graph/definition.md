<!-- process_media_graph -->

<!-- directory structure -->
.
├── definition.md
├── graph.py
├── __init__.py
└── utils
    ├── configuration.py
    ├── context.py
    ├── helper_functions.py
    ├── __init__.py
    ├── prompts.py
    ├── nodes.py
    ├── state.py
    └── tools.py

2 directories, 10 files

I need a data processing processing subgraph that will perform the following functions:

identify media based on type from the state ( conversational chat using HumanMessage, AIMessage classes alternating)

class GlobalState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # enables append/update

the message is sent from human (any document/media/url) with text that is a potential description of the media that triggers tool calling. This subgraph exists within a tool to process media in a larger graph with the same state and context. The entrypoint is a node "identify media" this will identify the media type to be processed. Media may include audio, video, images, text, documents, or urls of any of the previous. There are tools for each of the different types of media to handle their respective processing. All media types distil into text.

There is a tool for psycho-analysis of text using an llm with a structured output and a prompt.
There is a tool for emotional-analysis of text using an llm with a structured output and a prompt.
There is a tool for identifying and structuring relationships between the target and others in the text with a prompt and formatted output.

The features that are identified through tool use are used to update the assistant's identity.

[crawling bookmarks](https://claude.ai/chat/107e5a1b-96fb-4d31-b35b-a55080bda861)

[text and more](https://claude.ai/chat/30c554c8-1386-4af2-9f19-f63b51942fc5)

[process muliple media from a zip file or uploaded folder](https://claude.ai/chat/880c431e-bd37-4a2a-905a-b637369c60bc)