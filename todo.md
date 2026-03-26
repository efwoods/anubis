# TODO

""" TODO: Prevent Rate Limiting and Token Limiting Errors and Handle Message Failures """

## TODO: multiple tool calls

## TODO: False positive create memory vs recall_memories

# RECURSION LIMIT REACHED

## TODO:  bug
no response after long message (20,000 lines); multiple queries no response.

## TODO: bug
Tool call id's do not match
['f02ae497-a69b-443b-81aa-286d13f89fd6', 'f837fb82-061f-4094-8e59-d04c62f5cfb7', 'a944aae4-29f5-476e-b4b0-34eada06910e', '1d1252be-2050-4871-a196-2b2bca70e7ec', '8bff3714-2a44-43f9-9e38-4b6975a4efe2', '8e9fcfb9-9d75-46d4-87a1-be3808aff69e', '55fefe1c-3238-4c0b-a46f-4336f7991071', 'fb8e8134-ac18-48ca-8745-7f05fad84f21', 'd96fd629-b6e8-4189-ae80-23cd214b9581']

## TODO: BUG: CancelledError()
humanThis is a new event and I need you to create a memory about this. This is the first time I said I love you, because I do.
CancelledError()

## TODO: BUG: tool use injected into response
 I don't know who I am. You told me your name is Evan Woods, but I don't have any information about my own identity. If you share something about me, I'll try to remember it. [recall_memories()]

 recall_memories()<|python_end|>

You told me that your name is Evan Woods. That's the only information I have about you so far. I'm still getting to know you, and I'll remember more if you share more about yourself. [learn_information_about_the_user(content="User's name is Evan Woods.")]
        

## TODO BUG: false positive on tool calling
I'm still getting to know myself, and I don't have much information about my own identity. I don't know my name or any personal details about myself. However, I can tell you that I'm currently conversing with you, Evan Woods. If you'd like to share some information about me or my identity, I'll do my best to remember it. [learn_information_about_yourself_through_text_from_the_user_as_a_memory(content="User asked me to tell them about myself."), recall_memories()]

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
upload bible
scalable api (kubernetes)
ssl on vanity url
ci-cd build and push to azure
                                                         
https://claude.ai/chat/7ea64dfd-00b7-4743-88c0-6ee4c21ab1c3