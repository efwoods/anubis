# NAMESPACE GUIDANCE

all media will be distilled from video, audio, and images into text if not text. 

I will need to store the text as documents with the text as page_content with metadata on the document. 

I will use Langgraph document loaders from https://docs.langchain.com/oss/python/integrations/document_loaders/index#document-loader-integrations 

Everything is processed into a document. 

facts about the individual are stored in the vectorstore. 

Direct quotes are stored in the vectorstore. 

each quote is stored as its own document with metadata (page_content is the embedded field. notice the embedding field in @langgraph.json) 

So urls need to be parsed appropriately using document loaders into documents. 
Markdown files and text documents also need to be processed appropriately using document loaders.  
For videos and audio: videos will be processed into audio and diarized text and audio into diarized text. 

The diarized text where what was said directly from the individual during a dialogue is stored as a direct quote. 

The entire conversation of a dialogue itself is stored under the adapter namespace. 

For direct quotes without a dialogue or multi-user, a generated question prompt for each tweet if a series of tweets is generated for each tweet. If a monologue or presentation is the classification then a generated question prompt is created and the generated dialogue (synthetic question and original monologue or presentation) is stored in the adapter namespace and what was directly said (the original monologue or presentation) is stored as a direct quote. 

In the case of a dialogue (non-generated) the entire dialogue without any generation itself is saved, or the multi-user conversation saved, under the adapter namespace where the target identified avatar is converted to the label "assistant" and all other speakers are converted to the label "user". 

Only what was said from the target is stored as direct quotes in the quote namespace as documents. Only facts are stored about the identity of the individual are stored as documents in the identity namespace.

Dialogues containing other speakers cannot be stored as documents in the document namespace, the direct quotes need to be extracted from the dialogue of only the target avatar and stored in the direct quote namespace and the entire conversation of the dialogue is stored in the adapter namespace if the content is classified as a dialogue for adapter training at a later time. 

The namespaces in the vectorstore currently include: quote, document, memory, identity_memory, reference_image, reference_audio, and identity. 

Namespace usage:
identity: include documents about the individual (biographical information rewritten with an llm with facts preserved to prevent lawsuits [I need a prompt, class with structured output and class for intitialization and fact rewritting for this purpose to identify and extract and rephrase all facts without losing any information or hallucinating any information; use @ContentSituationClassificationClass for guidance and @schema for examples] and statements made about the individual rewritten in first person [all information in this namespace is first person perspective (I need a class for structured output, prompt, and class to initialize and rewrite facts into first person; use @ContentSituationClassificationClass for guidance and @schema for examples)], descriptions of the individual from images are stored in this namespace) (these are all primary sources of information)

quote: monologue, presentation, tweets, direct quotes, extracted quotes from dialogues or multi-user conversations, audio transcriptions directly from the individual are stored in this namespace

reference_image: contains a description of the individual and the base64 encoded string of the image (only one person in the image)

reference_audio: contains the text spoken of the individual and the base64 encoded string of the audio

document: these are reference documents that are not identity related but are used for RAG of the responses (menus for waiters, bible from pastors, proprietary content not used for identity )

memory: these are created during messaging conversation. They include surprising events and other miscellaneous memories created during conversation. 

identity_memory: these are created during messaging conversation and are secondary source information (the user told me that I am this attribute or have this quality).

I will be creating three avatars.

Pastor using @data/bible to be labled as proprietary content and stored in the document namespace
Waiter using @data/menu to convert the images into text then label the text as proprietary content and stored in the document namespace
Mom using @data/mom 

(non-actionable note: Dad avatar will have a data-analysis agent with a slack integration for sleep analysis and querying and alerts; 

Evan avatar will have a data-analysis agent feature with a slack integration for Afterlife Systems Situational awareness of the business (Sprint Planning, financial health, blockers, customer feedback, etc. ) and personal metrics (financial health, nutritional health, sleep health, exercise metrics) and social media subscription and logins)