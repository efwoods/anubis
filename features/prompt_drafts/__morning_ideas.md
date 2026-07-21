# Integrate metering; use with free tier (any non-subscriber goes through free tier); create discord voice agent; upload audio file to clone and use voice;

# MAVERICK 4 BASE MODEL WITH TRAINED ADAPTER USING GRPO RUNPOD FOR LEX FRIDMAN CHATBOT DATASET (PREPROCESS VIDEO TO PROMPT COMPLETION FORMAT) this will establish a baseline for the general public (preprocessing)
# preprocess social media accounts
# metering for three tiers
# testing of usage (free usage is message only; pro allows for upload of media to an extent)
# deterministic integrations (slack, discord, mcp, twitch bot for dm responses)
# pull one social media account, watch for a post, pull another media account

# MCP AND DATA ANALYSIS TO WATCH FOR HEALTH KIT DATA AND CREATE A HEALTH REPORT; QUERY THROUGH DISCORD, SLACK, UI, TWITCH

How much compute is required for GPU training vs inference?

# QUESTIONS
HOW MUCH DATA (PROMPT COMPLETION FORMAT) IS REQUIRED TO TRAIN AN ADAPTER (PROMPT-COMPLETION PAIRS):
WHAT IS THE IMPROVEMENT OF QUALITY OVER THE BASEMODEL: 
WHAT IS THE SIZE OF THE TRAINING DATA STORED (MB):

WHAT IS THE COST TO CREATE THE DATASET: 
  DIARIZATION: $2.50/1M TOKENS INPUT AND $10/1M TOKENS OUTPUT
  LLM ADJUDICATION: 
    $0.20/1M TOKENS INPUT AND $1.25/1M TOKENS OUTPUT
  DIARIZATION PER VIDEO SIZE:
  LLM ADJUDICATION PER VIDEO SIZE:


WHAT IS THE COST OF TRAINING THE ADAPTER:
HOW LONG DOES IT TAKE TO TRAIN AN ADAPTER:

WHAT IS THE COST OF INFERENCE WITH THE ADAPTER:


WHAT IS THE COST OF HOUSING THE BASEMODEL: 

Total per month Maverick 4 quantized 4 bit: 202.5 GB
250 GB
$17.50/mo

$0.07 per GB
---
Total per month Maverick 4 base model: 800 GB
1000 GB
$70.00/mo

----

<!-- 

# Adapter compute per hour: 

Runpod Pytorch 2.8.0
3x A100 PCIe
240GB VRAM · 375 GB RAM · 36 vCPU
Total / hour
$4.21 USD
Billed per millisecond
Total disk
300 GB
GPU cost
$4.17/hour
Container disk cost
$0.035/hour
Volume disk cost
$0.007/hour
Stopped cost
$0.014/hour 

# model storage: 
Amazon S3 pricing is heavily dependent on the specific storage class you select for your 250GB:S3 Standard (Frequent Access): ~$0.022 per GB → ~$5.50 / month

IDLE COST $0.33/day

-->


WHAT IS THE TIME TO FIRST TOKEN WHEN USING THE ADAPTER:

WHAT IS THE COST OF STORING THE ADAPTER:

HOW MANY PEOPLE CAN BE SERVED INFERENCE SIMULTANEOUSLY:

# To develop:
- algorithm to switch from basemodel and visa versa
- algorithm to detect when their is sufficient data to train
- algorithm to store the data and adapter

# hide unused api endpoints

# coalesce and use select avatar into single endpoints

# QR code geo located per use case to share, onboard, and connect with application use cases (deep research on the individual to identify accounts, geo located per business to integrate with the business)

# User logs in: new user detected with facial recognition, walks through onboarding, scans for connected accounts, asks for permission for integration, informs how the avatar may triage requests on your behalf. 
- Centralized interface for notifications from all platforms 
  - (responses required; 
  - ongoing conversations and messages handled on your behalf; 
  - personal analytics dashboard; 
  - queryable interface for:
    - dm with self to ask about the current state of affairs of anything, 
    - notifications of the following
    - handle any request, 
    - answer any unhandled message, 
    - view for avatar documents for identity/ update manually)
  - switch to any avatar
  - settings and connect accounts



# Better UI: Login
# Improvement of response quality
# Geo located Avatars (memorials real-world with basic facts)
# Adapters vLLM server Llama possibly

<!-- - [x] I need to chunk the logs -->
- [ ] I need to work on git_log_all.sh additional update to the log file

- [x] I need to build a stylistic profile
- [x] I need diarization completed
- [ ] I need metering of responses

- [x] I need summarization middleware
- [x] I need to test using a deep-agent in lieu of a model
- [ ] I need to work on analysis (IMAP, playwright login, pull)

- [ ] Generative query retrieval to improve responses

drag and drop zip files and have all the media processed
<!-- send urls -->
data analysis/slack bot

add and process a list of urls and zip files

authenticity score against baseline chatgpt output (how unalike is the content style from base chatgpt; then after a threshold of ground truth, how alike is the content to the ground truth data?)

STREAMLIT: USE FACIAL RECOGNITION/VOICE RECOGNITION FOR LOGIN RATHER THAN TRADITIONAL EMAIL SYSTEMS (PERMISSION IS GRANTED FROM THE WEBSITE TO USE CAMERA AND MIC; THERE IS AN AVATAR THAT COMMUNICATES PRIVACY AND TERMS OF SERVICE;  A REFERENCE IMAGE IS CAPTURED, REFERENCE AUDIO IS CAPTURED; THIS IS USED IN LIEU FOR LOGIN and signup)

----

# Frontend response should be cached and hidden unless the response completes before a message is sent; otherwise the message request is cancelled in lieu of the user's sent response; The user may type immediately, there is a "thinking" indicator; 

Hey! Please tell me about yourself and what you can do for me.
assistant avatar

Hey! I’m Evan Woods, the founder of Neural Nexus.

At a high level, Neural Nexus is a messaging platform where you can create custom large language models and use them for neural-interface style communication. The core idea is: you bring in text (and optionally other signals), and we help you train a custom LLM so it can respond in a writing style that’s “akin to someone you love.” That can be for nostalgic/therapeutic conversations—like talking with a loved one who’s passed—or for fun/entertaining interactions with someone who wouldn’t normally respond.

On top of that, we’re building novel input/output paths. For example:

    Text + image support: you can send images along with text, including conventional image sending.
    Audio transcription for context: the system can listen to short audio chunks (on the order of a few seconds) and transcribe them to improve conversation suggestions.
    Thought-to-image reconstruction (research direction): there’s a pipeline that takes an image and reconstructs it via synthetic waveform generation and reconstruction steps.
    Custom LLM integration: we preprocess text messages into a dictionary, train a large language model, and use retrieval/context from our stack (embedding model + vectorstore + Postgres) to ground responses.

What I can do for you depends on what you’re trying to build or solve. A few common directions:

    Help you design a custom “voice” model (what data to use, how to structure prompts, how to think about style consistency).
    Walk you through how retrieval/context works in the system (embedding → vectorstore → Postgres → context injection).
    Brainstorm use cases (therapy/nostalgia, accessibility, entertainment, etc.).
    If you’re working on an implementation, I can help you map requirements to architecture and suggest a clean pipeline.

If you tell me what you want to accomplish—like “I want a model that writes like X,” or “I want to add image/audio input,” or “I’m building the backend”—I’ll tailor the next steps.

# upload media
add optional context to media upload 
allow the process media graph to be used as a tool triggered in conversation

# Evaluation
# Tuning Prompts to reduce false negatives
# Tuning Prompts to reduce false positives
# Tuning Behaviors in Threads
# Train-Attach-Evaluate Adapter Lifecycle

# Scan and pull data from scripts after setting an explicit target
# Scan and parse quotes and biographical information from websites (curie)
# Increase quality of data from pdf, random websites, markdown
# crawl linktree;
# recurse media sites
# Deep research

# get reference audio file and reference image file url endpoints
# CRUD memories/identity_memories endpoints

# I need avatar_id as a parameter for list_avatar_documents; delete_avatar_documents; use selected avatar if avatar_id is not present: verify that the requester is the creator in the endpoint

# Moving forward

# measure quality
# create and attach adapters
# create integrations to social media platforms
# Create integrations to communications platforms
# Ambient email agent has information on updating preferences; triaging
# MCP Server to access data as an integration
# save transcriptions for parsing from calls or video meetings

# MCP servers for connecting accounts and pulling information


------
# 
# geo located assistant (each ) for ordering creates "fast-pass" experience to skip lines and enter orders via llm into a queue for local fast food restaurants and services; 
# twitch moderators/personal avatars for personalized responses (@ev0ra_bot) to receive a response akin to and based on the likeness of that person
# frontend experience with a gallery view of avatars; use facial recognition login, communicate privacy policy and terms of service, interesting visualization, capture reference image and reference audio, onboarding process -> create your own avatar, suggest integrations (slack, discord, twitch, MCP for data analysis) select if your avatar is private or public (share your avatar with the world!). Create your own avatar, select a variety of avatars to communicate with (public avatars). 
# Video game/fictional characters (realtime-interaction using audio files to establish the experience of creation of the entity and translate that same experience to other real-people). communicate with the likeness (generate audio profiles etc.) based on the available data.
# Augmented reality full-body avatars; neural data driven media;