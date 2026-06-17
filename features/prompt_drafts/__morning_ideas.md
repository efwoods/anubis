# Better UI: Login
# Improvement of response quality
# Geo located Avatars (memorials real-world with basic facts)
# Adapters vLLM server Llama possibly

<!-- - [x] I need to chunk the logs -->
- [ ] I need to work on git_log_all.sh additional update to the log file

- [ ] I need to build a stylistic profile
- [ ] I need diarization completed
- [ ] I need metering of responses

- [ ] I need summarization middleware
- [ ] I need to test using a deep-agent in lieu of a model
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

# Measure Authenticity
authenticity score against baseline chatgpt output (how unalike is the content style from base chatgpt; then after a threshold of ground truth, how alike is the content to the ground truth data?) (use a baseline of questions, generate responses to the baseline of questions and compare)


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

# I need avatar_id as a parameter for list_avatar_documents; delete_avatar_documents; use selected avatar if avatar_id is not present; verify that the requester is the creator

# Moving forward

# measure quality
# create and attach adapters
# create integrations to social media platforms
# Create integrations to communications platforms
# Ambient email agent has information on updating preferences; triaging
# MCP Server to access data as an integration
# save transcriptions for parsing from calls or video meetings

