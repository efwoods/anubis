# Claude project url: https://claude.ai/chat/910f48f7-dc1b-4fe9-9ffe-169c74794ea6

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


# Currently have (in process sequence)

## (STEP 1) how i am accepting media (files and urls)
@app.post("/update_avatar_identity_with_media")
async def update_avatar_identity_with_media(
    files: Optional[List[UploadFile]] = File(...),
    url: Optional[str] = None,
    assistant_id: str = None,
    reference_audio: bool = False,
    reference_image: bool = False, 
    proprietary_content: bool = False, 
    current_user: dict = Depends(get_current_user)
):
    # Context user_id, assistant_id
    logger.info(f"UPLOAD MEDIA ENDPOINT ENTRY")
    """
    Upload one or more media files for processing and indexing.
    
    - **files**: One or more files to process
    - **user_id**: User identifier
    - **assistant_id**: Assistant identifier
    """
    try:

        user_id = current_user['identities'][0]['user_id']

        # assitant_config = current_user['app_metadata']['assistant_config']
        # assistant_id = assitant_config['configurable']['assistant_id']  
        # config['configurable'].update(assitant_config['configurable'])
        
        config = {
            "configurable": {
                "user_id": user_id,
                "user_ctx": {"name":None, "description": None},
            }
        }

        config['configurable']['assistant_id'] = assistant_id
        config['configurable']['assistant_ctx'] = {"name":None, "description": None},
        # Read all uploaded files
        media_files = []
        for file in files:
            content = await file.read()
            media_files.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": reference_audio,
                "reference_image": reference_image,               
                "proprietary_content": proprietary_content
            })



## (STEP 6) Create questions from ground truth statements (strings only; will need to process json) src/anubis/utils/dataset/formatting.py

async def create_question_list(str_messages_list: list[str]) -> List[str]:
    class GeneratedQuestionsList(BaseModel):
        question_list: List[str]

    human_messages_list = [HumanMessage(content=message_str) for message_str in str_messages_list]

    model_with_structured_output = init_model(model_without_tools=False, response_format= GeneratedQuestionsList)

    system_message = SystemMessage(content="Given this list of messages, generate a query to which the message is the response. THERE MUST BE A QUESTION FOR RESPONSE AND THE QUESTION ORDER IN THE LIST MUST MATCH THE RESPONSE ORDER. These questions must be succinct.")
    
    messages = [system_message] + human_messages_list

    response = await model_with_structured_output.ainvoke(input=messages)

    return response.question_list


## (STEP 4 & STEP 6) Creating adapter training format from question answer pairs (strings only will need a driver to process json) src/anubis/utils/dataset/formatting.py

""" LLAMA 4 ADAPTER TRAINING FORMAT """

async def llm_single_turn_dataset(question_list: List[str], answer_list: List[str]) -> List[dict]:
    """ Creates a Messages Dataset of Single Turns for a list of question and answer pairs. Used for LLM Adapter Training Format."""
    single_turn_dataset = []
    for question, answer in zip(question_list, answer_list):
        turn = {"messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]}
        single_turn_dataset.append(turn)
    return single_turn_dataset

def llm_multiturn_dataset_one_conversation(question_list: List[str], answer_list: List[str]) -> dict:
    """ Creates a Messages Dataset of a conversation of question and answer pairs. Used for LLM Adapter Training Format.
        This is a single conversation. A list of multiple conversations must be used to for the entire final dataset.
    """
    list_of_messages = []
    for question, answer in zip(question_list, answer_list):
        turn = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        list_of_messages += turn
    multi_turn_dataset = {"messages": list_of_messages}
    return multi_turn_dataset

""" LANGSMITH DATASET FORMAT """

async def langsmith_dataset(question_list: List[str], answer_list: List[str], dataset_source_filename: str) -> List[dict]:
    """ Creates a list of dict example question and answer inputs and outputs """
    examples = []
    examples.append({
        "inputs":{"question": question}, 
        "outputs": {"answer":answer}, 
        "metadata": {"source", dataset_source_filename}} for question, answer in zip(question_list, answer_list))
    return examples


###########
# EXAMPLE OF MODEL WITH STRUCTURED OUTPUT
Examples of analysis models with structured output:
class TextualSituationalAwareness(BaseModel):
    classified_situation: Literal["single_speaker", "q_and_a_dialogue", "multi_speaker", "other"]
    reasoning: str = Field(
        description = "Step-by-step reasoning behind the decision for the classified situation of the text. (single speaker monologue, single tweet from user, strictly Q & A, multi-speaker, Other)"
    )



TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS = """
<Role>
Your role is to analyze and classify text with respect to the situation of the content within the text.
</Role>

<Instructions>
Your objective is the following:
Classify the text and decide whether the text contains one of the following situations:
- A single speaker
- Strictly question and answer between two speakers in a dialogue
- Multiple speakers
- Other 

Present a clear succinct reason why the classification was chosen using examples from the source text to support your reasoning.
</Instructions>

<Rules>
=========== SINGLE SPEAKER GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for single speaker situations:

Classify the text as a single speaker given text in the following situations:
- There is a single tweet
- There is a single statement
- There is a label of the speaker and there is only one speaker
- There is only a single speaker detected in the content

Use the following examples to help determine the situation of the given text for single speaker situations:

Example Tweet or single speaker statement:
I believe that through the research and development of A.I., we will understand what is most valuable about being human.

=========== QUESTION AND ANSWER DIALOGUE GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for question and answer dialogue situations:

Classify the text as strictly question and answer between two speakers in a dialogue:
- There is more than one speaker but less than three speakers in the text
- There is turn-taking between two speakers
- There are labels of the speakers and there are only two speakers

=========== MULITPLE SPEAKER GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for multiple speaker situations:

Classify the text as multiple speakrs:
- There is more than one speaker but less than three speakers in the text
- There is turn-taking between two speakers
- There are labels of the speakers and there are only two speakers

=========== OTHER GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for situations that do not appropriately match and of the situations for single speakers, question and answer dialogue, or multiple speakers. 

Classify the text as other:
- The criteria for single speaker, question and answer dialogue, and multiple speakers do not apply

Rules for classifying the situation of the text as Other:
- This should not often be used and is treated as an unclassified error and failure in analysis
- This should almost never be used
- There needs to be an explicit reason why this is classified as other, why the single speaker classification does not match, why the question and answer dialogue does not match, why the multiple speaker classification does not match, and each reason needs to be explicitly listed for each reason why the previous classifications did not match the criteria.
- There needs a suggested classification for this situation with an explicit reason why this new suggested classification appropriately classifies the situation of the text better than any of the other reasons.
- Try not to classify any situation of the text as Other.
</Rules>
"""

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