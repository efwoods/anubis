
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