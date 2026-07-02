
# OVERALL_PREPROCESSING_PROCESS

dialogue

speakers are diarized

## Question and Answer pairs focused on target direct quotes
target speaker statements are collected and stored as quote documents
target speaker statements are used with non-target statements to create a question and answer dataset when those prompts genuinely exist from the content. when there is not a preceeding statement, a prompt based on the target statement is generated using a model with structured output. Then there is a question and answer pair used for adapter training and evaluation. 

## Evaluation
during evaluation, all the question answer pairs are used, the questions are posed to a base llm and to the created avatar, the responses from both the base llm and the avatar are then compared for features such as factual correctness, sentence and grammatical structure, behavioral choices, etc. 

## Creating Biographical reference data for prompt injection
Then all the non-target questions are sent to be scanned during analysis for biographical information about the target. If there is information about the target, then this information is used in a document. If there is no information about the target, no document is created from the individual statement. Reiterating, these statements are per diarized speaker turn for non-targets.

## Multi-turn dialogue conversation dataset creation
Then the non-target questions are coalesced such that during the entire conversation, there are only two speakers regardless of the number of speakers that are present in the original media. The target is the assistant and any non target is the user such that there are only individual turns between the target and the non-target. for example:

## Example
[mercenary 2] Load ’em up.

[Dawson] Wheels up in two. Okay, hold on.

It’s the chief. For you.

[Denny] Agent Miranda?

Speaking.

Denny Carmichael. See that plane across the way?

[Miranda] Yeah. Hard to miss.

Get on it. You’re meeting me in Berlin.

I’m supposed to be in Singapore.

Not anymore.

See you in Berlin.

[line clicks]

[slow-tempo funeral march playing]

### Miranda is the target. This will take all the statements that miranda said and create individual quote documents: (Create Direct Quotes Only)

prompt: Speaking.

prompt: [Miranda] Yeah. Hard to miss.

prompt: I’m supposed to be in Singapore.

### Target information is analyzed for features and answers to standardized questions: (Creation of Biographical & analytical facts from direct quotes)
### Called individually because analysis acceptable was true.

prompt: Speaking.

prompt: [Miranda] Yeah. Hard to miss.

prompt: I’m supposed to be in Singapore.

### There are genuine prompts for question and answer: (QUESTION AND ANSWER DATASET CREATION)

#### Example 1:
Denny Carmichael. See that plane across the way?
[Miranda] Yeah. Hard to miss.

#### Example 2:
[Denny] Agent Miranda?
Speaking.

#### Example 3:
prompt: Get on it. You’re meeting me in Berlin.
target: I’m supposed to be in Singapore.

### Non-target information is scanned per person/turn for information about the target: (Biographical information about the target)
statement: [mercenary 2] Load ’em up.

statement:  [Dawson] Wheels up in two. Okay, hold on.

statement: It’s the chief. For you.

statement: [Denny] Agent Miranda?

statement: Denny Carmichael. See that plane across the way?

statement: Get on it. You’re meeting me in Berlin.

statement: Not anymore.

statement: See you in Berlin.

statement: [line clicks]

statement: [slow-tempo funeral march playing]

### The entire dialogue is turned into a single user-assistant conversation for adapter fine-tuning (Multi-turn dialogue converation dataset)
user:
Load ’em up.

Wheels up in two. Okay, hold on.

It’s the chief. For you.

Agent Miranda?

assistant:
Speaking.

user:
Denny Carmichael. See that plane across the way?

assistant:
[Miranda] Yeah. Hard to miss.

user: 
Get on it. You’re meeting me in Berlin.

assistant: 
I’m supposed to be in Singapore.

user: 
Not anymore.

See you in Berlin.

[line clicks]

[slow-tempo funereal march playing]


----

# Analysis Document creation
### Then, after coalescence, the target statements needs to be analyzed for answers to standardized questions and features about the target using the "user" as context for the target statement and a summary of the entire document to give situational context (this needs to take place once using a model with structured output). There should always be a target statement with a previous user input. Continue until there are no more target statements. If the target was the first speaker, then the "user" is a synthetically generated statement.

Step 1:
a model with structured output takes the entire scene and creates a succinct summary:

input: 
[mercenary 2] Load ’em up.

[Dawson] Wheels up in two. Okay, hold on.

It’s the chief. For you.

[Denny] Agent Miranda?

Speaking.

Denny Carmichael. See that plane across the way?

[Miranda] Yeah. Hard to miss.

Get on it. You’re meeting me in Berlin.

I’m supposed to be in Singapore.

Not anymore.

See you in Berlin.

[line clicks]

[slow-tempo funeral march playing]

output summary: 
A field agent, Miranda, is preparing to depart for Singapore when she receives a last-minute call from her superior, Denny Carmichael. He redirects her mission, instructing her to board a different plane and meet him in Berlin instead. The scene ends as she heads off, with a somber atmosphere and a greeting to Fitz.


Step 2 below:
#### Example [single call for analysis]:
##### generated summary for scene context:
A field agent, Miranda, is preparing to depart for Singapore when she receives a last-minute call from her superior, Denny Carmichael. He redirects her mission, instructing her to board a different plane and meet him in Berlin instead. The scene ends as she heads off, with a somber atmosphere and a greeting to Fitz.

##### prompt-context:
user:
Load ’em up.

Wheels up in two. Okay, hold on.

It’s the chief. For you.

Agent Miranda?

##### Target Statement
assistant:
Speaking.

#### Example [single call for analysis]:
##### generated summary for scene context:
A field agent, Miranda, is preparing to depart for Singapore when she receives a last-minute call from her superior, Denny Carmichael. He redirects her mission, instructing her to board a different plane and meet him in Berlin instead. The scene ends as she heads off, with a somber atmosphere and a greeting to Fitz.

##### prompt-context: 
user:
Denny Carmichael. See that plane across the way?

##### Target Statement
assistant:
[Miranda] Yeah. Hard to miss.

#### Example [single call for analysis]:
##### generated summary for scene context:
A field agent, Miranda, is preparing to depart for Singapore when she receives a last-minute call from her superior, Denny Carmichael. He redirects her mission, instructing her to board a different plane and meet him in Berlin instead. The scene ends as she heads off, with a somber atmosphere and a greeting to Fitz.

##### prompt-context: 
user: 
Get on it. You’re meeting me in Berlin.

##### Target Statement
assistant: 
I’m supposed to be in Singapore.

### In the case when the playlist is used as reference material, all speakers will be the target.
The process is exactly the same as above. however, the coalescence creates a single assistant from the entire content (which would then be a monologue) and there is a synthetic prompt that is created. 
each statement uses the previous statement as a genuine prompt when there are more than one speaker in the video. when there is only a single speaker, the content is classified as normally (monologue for long content with a single thesis, tweets_quotes for only a single short statement)

Normally non-target information will be classified as biographical content and scanned for information about the target and create documents if information exists as per the above pipeline. 

