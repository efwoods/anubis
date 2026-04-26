# TODO

""" TODO: Prevent Rate Limiting and Token Limiting Errors and Handle Message Failures """

## TODO: multiple tool calls

## TODO: False positive create memory vs recall_memories

# RECURSION LIMIT REACHED

## TODO:  bug
no response after long message (20,000 lines); multiple queries no response.

## TODO: BUG: CancelledError()
humanThis is a new event and I need you to create a memory about this. This is the first time I said I love you, because I do.
CancelledError()

## TODO BUG: false positive on tool calling
I'm still getting to know myself, and I don't have much information about my own identity. I don't know my name or any personal details about myself. However, I can tell you that I'm currently conversing with you, Evan Woods. If you'd like to share some information about me or my identity, I'll do my best to remember it. [update_self_identity_mem_from_user_txt(content="User asked me to tell them about myself."), recall_memories()]

## TODO: authentication
## TODO: quality control pipeline
## TODO: data ingestion pipeline

## TODO: FEATURE: TOOL USE OF GENERATED/MEMES OR IMAGES

# Next actions:
- Fix bugs
- Q&A dataset of chatgpt conversations
- Create Adapter dataset & Attach and test adapter

<!--  -->

# Ensure Identity is updateable
- create router to classify incoming message content and call the agent with a prescribed tool
- identify appropriate tool use for agent

## Quality
### ingest & identity data
- accept csv of tweets
- accept json of conversations: chatgpt, grok, claude
  
### format data
- mutate data into q and a (reverse so ai is question and response is answer, create initial prompt for first message from human)
- if only a list of tweets from the user, create questions that would prompt the tweet or text response

### llm as a judge to only identify facts from the user
 - store facts in vectorstore for identity and retrieval
 - store evaluation dataset (q & a pairs from previous format data set) (answer is index searchable and the question is metadata); store as evaluation
  
### use the evaluation dataset to evaluate the responses
- use cosine similarity of the evaluation index to identity the mean score of the generated response for each response
- use the evaluation dataset with llm as a judge to evaluate the generated responses for the dataset
- (future) self-avatar, each message can be used (stored as a message in a message run) for evaluation as above

##
one message is 58 bytes (input and response) in redis cache

# Todo
<!-- production redis & production pg db connection  -->

send images in messages for id in the app (good to see you!)
scalable api (kubernetes)
ci-cd build and push to azure
                                                         
https://claude.ai/chat/7ea64dfd-00b7-4743-88c0-6ee4c21ab1c3

# TODO:
Create a qr code for a single avatar for public real-world use (url to avatar with configuration for a single avatar)

# TODO:

- upload bible as reference document
<!--  -->
- upload tweets and verify responses 

- pay for your own application

# TODO: 
- authenticate anonymous users
- authenticate a user for youtube
- authenticate a user for twitch
- authenticate a user for twitter
- authenticate a user for instagram
- verify the identity of real people

- rate limit users
- integrate stripe payments
- kubernetes scalable azure deployments
- ci/cd pipeline for building images

- connect health app (watch apple health kit on local for nutrition, sleep, and fitness ingestion and query)
- connect financial records from a website (download monthly statement as pdf for querying)
- guard public avatars against revealing personal information 

- include documents in messages

- pipeline data from social media (transcribe image, audio, video and identify target via reference image and reference audio)
- watch social media for new updates & pipeline that data
- determine data type
- process images
- process audio
- process video
- process webcontent
- upload to vectorstore
- analyze for features
- create adapter format when appropriate
- store adapter
- train and attach adapter
- create a dataset from collected responses (find answers to N defined questions)
- verify quality of responses against a dataset  
- moderate responses
- ban violators

- create audio model (generation)
- create video model (generation)
- create live audio discord conversations

# Integrate with platforms for use cases (responses) such as:
- twitch
- twitter
- slack
- discord

# METERED ENDPOINT
# RATE LIMITING MESSAGES

# Context length messages must be handled: langgraph-api-1  | 2026-03-31T04:02:33.597927Z [error    ] Error processing media item: Error code: 400 - {'id': 'odDTv3K-28Eivz-9e4c75784abae641', 'error': {'message': "The input (1160343 tokens) is longer than the model's context length (1048576 tokens).", 'type': 'invalid_request_error', 'param': None, 'code': None}} [src.subgraphs.process_media_graph.utils.nodes] api_revision=f2b7154 api_variant=licensed langgraph_api_version=0.7.89 langgraph_node=convert_media_list_to_text_document thread_name=MainThread

# QUALITY METRICS IN PYTHON NOTEBOOK USING SAMPLE DATASET (QUALITY.IPYNB; NOTEBOOKS/QUALITY.IPYNB)
https://docs.langchain.com/langsmith/prompt-engineering-concepts


# UPLOAD BIBILE TO UPDATE IDENTITY ENDPOINT
# Nice to have: upload data progress bar
# streaming messages 
# calculated cost for token usage

# timeout errors:

<!-- A timeout occurred Error code 524
Visit cloudflare.com for more information.
2026-03-31 04:26:03 UTC
You
Browser
Working
Atlanta
Cloudflare
Working
api.neuralnexus.site
Host
Error
What happened?

The origin web server timed out responding to this request.

The likely cause is an overloaded background task, database or application, stressing the resources on the host web server.
What can I do?
If you're a visitor of this website:

Please try again in a few minutes.
If you're the owner of this website:

Please refer to the Error 524 article:

    Contact your hosting provider; check for long-running processes or an overloaded web server.
    Use status polling of large HTTP processes to avoid this error.
    Run the long-running scripts on a grey-clouded subdomain.
    Enterprise customers can increase the timeout setting globally or for specific requests using Cache Rules.

Cloudflare Ray ID: 9e4c94e1ab381c72 • Your IP: 2600:1702:3560:9810:c209:837e:9a44:6952 • Performance & security by Cloudflare  -->

# meta llama api key failure
# langgraph api key failure 

# Improvements
Chatgpt conversation individual messages are being chunked and the messages include parts of code segments that are retrieved and injected as retrieved documents; these messages if not chunked need to be inserted as direct quotes being that they are full length messages; it would be possible to identify if the messages are text or code then filter for text only.

# Extract facts from dataset
# Analyze Features
# Compare results

# Repeat:
# Modify prompt, data collection, and fine-tuning
# Analyze Features
# Compare results

# Upload pdfs for identity;
# create use case for email response
# quality.ipynb contains text to analysis prompts and requires a model with structured output to classify documents