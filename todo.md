# TODO

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