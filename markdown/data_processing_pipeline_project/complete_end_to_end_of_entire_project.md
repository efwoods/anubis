# Prereq:
- individuals authenticate with social media
- social media is crawled for content for the pipeline initially
- social media is subscribed to for updates which are pulled and pipelined
- users are identified for personal avatars with name, phone number, address, 1 social media account, and email (limit 1 per user)
- deep research researches facts about the target and pipelines the data as below
- human verification of facts and source

# Full Data collection and processing pipeline process overview:

1. The content type needs to be identified (mime type) then converted into text:

2. The content of any text document needs to be identified for situation: (facts about a user from a biographical website or conversational media: dialogue, monologue, tweets, presentation from a user )

3. Then the text if conversational needs to be converted to the following:

message format (narration needs to be a user defined as narrator)

first all media needs to be converted to the following format if not facts (if a dialogue, monologue, tweets, presentation from a user, etc.):

"messages": [
      {
        "speaker": "narrator",
        "content": "stylistic message about environment, situation, or world"
      },
      {
        "speaker": "NAMED_SPEAKER",
        "content": "What the user is saying. Actions of the user are in asterisks: e.g. *I hug you*; *I smile fondly*"
      }
    ],

4. Then, the above is converted to the following format:
final format after target identification where all other individuals are "user" and target is "assistant";

  "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant that provides concise answers."
        },
        {
          "role": "user",
          "content": "What is the capital of France?"
        }
      ],

5. Then all messages from the target are analyzed for content, all messages from the target are stored themselves for retrieval.

6. Individual messages from a user such as a monologue or presentation or a series of tweets have generated questions to prompt the ground truth messages as responses and the format is as previously mentioned: (i.e.:  "messages": [{"role": "system", "content": "this is the content"}, {"role":"user", "content":"what the target is saying"}])

7. the conversational messaging in a dialogue (whether ground truth or generated synthetic prompting questions) is saved for adapter training. (Future increase in quality: after sufficient data and time period, the data is used to train an adapter, store an adatper, and attach the adapter when the avatar is selected if the adapter is available.)

#########################


# How the data is then used 
1. The model is queried or an initial query is created
2. situational awareness is loaded for identity, memories, knowledge, and the analyzed facts and relevant previous statements into a system prompt
3. an adapter is attached if available
4. text or other media is generated

# use cases
- prayers
- email responses
- twitter bots
- twitch bots
- discord bots
- real-time audio
- waiters
- memoirs
- geographical informational markers with interactive messaging
- generated media
- vr genie3 world models
- thought to text text to thought with neuralink
- git commit messages that are custom
- personal motivational speakers
- self awareness through psycho analysis
- connecting to health data and fitness data and financial data for self awareness and promotion of well being

# Future tuning and improvement of choices ( the responses must be evaluated for quality and preferences must be noted and prompt injected as examples and traces must be rewardes as examples of correct versus incorrect behavior choices)

# Immediately available
custom commits
text responses
factual self awareness from uploaded media
menus and ordering for restaurants
prayers