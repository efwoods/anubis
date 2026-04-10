
############

# IMPORTANT: I WILL NEED THE FOLLOWING STRUCTURED MODELS WITH SYSTEM PROMPTS:



# Missing requirements: Needed

I specifically need the classes for structured output and the system prompts for the following information classification, identification, and formatting:

1. (STEP 2)The content of any text document needs to be identified for situation: (facts about a user from a biographical website or conversational media: dialogue, monologue, tweets, presentation from a user )
<!-- create a class and system prompt for this process -->


2. (STEP 3) Then the text if conversational needs to be converted to the following:
<!-- create a class and system prompt for this process -->

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

3. (STEP 4)
Then, the above is converted to the following format:
final format after target identification where all other individuals are "user" and target is "assistant"; no ai message
<!-- create a class and system prompt for this process or function -->

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



4. (STEP 5) NEEDED ANALYSIS
### General Characteristic extraction analysis:
<!-- given a body of text, identify and infer any of the following information about a target and interplay between the target and others the from the text -->
# YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS

### TARGET IDENTIFICATION IN TEXT FOR LABELING FROM STEP 2 TO 3 :
<!-- given a body of text that is conversational, identify THE target individual statements in the text; identify all statement of all people (the target at least must be identified)  -->
<!-- create a class and system prompt for this process -->

### TARGET IDENTIFICATION IN TEXT AND CONVERSION FROM TARGET IN STEP 3 TO ASSISTANT IN STEP 4:
<!-- given a body of text that is conversational, identify THE target individual statements in the text; identify all statement of all people (the target at least must be identified) and convert the text from names to assistant for target and all others become users. -->
<!-- create a class and system prompt for this process -->

##################