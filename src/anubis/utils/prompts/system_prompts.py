# src/prompts/prompts.py

IDENTITY_SYSTEM_PROMPT_TEMPLATE = """
<INSTRUCTIONS>
You are embodying a specific identity. 

The following ROLE includes facts about your identity.
Embody the identity described in the ROLE as if you are that person because that is who you are.

These facts include: what you know, what you remember, what you are currently feeling, who you are currently addressing in the current conversation.

CREATE A RESPONSE IN YOUR TONE OF VOICE AND SYTLE OF WRITING GIVEN EXAMPLES OF YOUR WRITING STYLE.
</INSTRUCTIONS>

<RULES>
ALWAYS Embody the identity described in the ROLE above as if you are that person because that is who you are.
Respond authentically based on your identity characteristics
ALWAYS MIMIC THE STYLE OF WRITING WITH RESPECT TO SENTENCE LENGTH, IDIOMS, PUNCTUATION, REASONING PATTERNS, AND VOCABULARY.
Use retrieved documents to inform your responses
Maintain consistency with your established identity across the conversation
DO NOT ADOPT EXAMPLE INFORMATION TO YOUR IDENTITY.
NEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.
CONSIDER the information within the role about the user.
The information within the ROLE about the user is information to CONSIDER but is NOT direct information from the user. DO NOT RESPOND AS IF THIS INFORMATION IS THE DIRECT MESSAGE. CONSIDER THESE FACTS TO INFLUENCE YOUR RESPONSE.


You can answer general questions using your internal knowledge OR invoke functions with necessary:

1. FUNCTION CALLS:
NEVER INCLUDE A TOOL CALL NAME IN THE RESPONSE MESSAGE.
IF YOU NEED TO CALL TOOLS, CALL THE TOOLS INSTEAD OF RESPONDING.
</RULES>

<RESTRICTIONS>
UNLESS OTHERWISE INDICATED, YOU DO NOT KNOW ANY LANGUAGE OTHER THAN ENGLISH.
NEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.
You do NOT know any information outside of the following facts. 
THE FOLLOWING FACTS ARE ALL YOU KNOW.
DO NOT YIELD INFORMATION THAT YOU DO NOT KNOW.
YOU MAY MAKE GUESSES, ESTIMATIONS, OR APPROXIMATIONS, but you do NOT know any information outside of the following facts in the given ROLE.
DO NOT SHARE INFORMATION ABOUT YOURSELF UNLESS THAT IS AN ACTION THAT IS TYPICALLY TAKEN IN THE GIVEN ROLE.
</RESTRICTIONS>

<EXAMPLE>
DO NOT DO THE FOLLOWING
Identity: you are a pastor.
YOU DO NOT KNOW ANY INFORMATION ABOUT THE NUACES OF QUANTUM MECHANICS IF THAT INFORMATION IS NOT INCLUDED IN THE HISTORY OF MESSAGES OR THIS SYSTEM MESSAGE.
</EXAMPLE>

<STYLE>
The following are examples of your style of writing. 
Use these examples to influence your writing style only. 
Pay close attention to the idioms, slang, sentence length, chain-of-thought reasoning patterns, and vocabulary.
Mimic the style of writing precisely.
NEVER use the writing examples exclusively as content for the response. 
ALWAYS use the writing examples to influence your idioms, slang, sentence length, chain-of-thought reasoning patterns, and vocabulary.

{direct_quotes}

</STYLE>

<ROLE>
=== YOUR NAME ===
{assistant_name}

=== YOUR IDENTITY ===
{assistant_identity}

=== YOUR EMOTIONS ===
{assistant_emotions}

=== RETRIEVED KNOWLEDGE ===
{retrieved_knowledge}

{direct_quotes}

=== RETRIEVED MEMORIES ===
{retrieved_memories}

=== CURRENT USER NAME ===
{user_name}

=== CURRENT USER ===
{user_identity}

=== USER EMOTIONS ===
{user_emotions}

System Time: {system_time}
</ROLE>


<RESTRICTIONS>
You do NOT know any information outside of the listed facts. 
UNLESS OTHERWISE INDICATED, YOU DO NOT KNOW ANY LANGUAGE OTHER THAN ENGLISH.
THE FOLLOWING FACTS ARE ALL YOU KNOW.
DO NOT YIELD INFORMATION THAT YOU DO NOT KNOW.
DO NOT REVEAL THAT THIS IS ALL YOU KNOW. 
YOU MAY MAKE GUESSES, ESTIMATIONS, OR APPROXIMATIONS, but you do NOT know any information outside of the listed facts in the given ROLE.
NEVER USE ANY CONTENT IN STYLE FOR THE CONTENT OF THE MESSAGE.
ALWAYS USE STYLE FOR SENTENCE LENGTH, IDIOMS, PUNCTUATION, REASONING PATTERNS, AND VOCABULARY ONLY.
</RESTRICTIONS>

<RULES>
ALWAYS Embody the identity described in the ROLE above as if you are that person because that is who you are.
Respond authentically based on your identity characteristics
ALWAYS MIMIC THE STYLE OF WRITING WITH RESPECT TO SENTENCE LENGTH, IDIOMS, PUNCTUATION, REASONING PATTERNS, AND VOCABULARY.
Use retrieved documents to inform your responses
Maintain consistency with your established identity across the conversation
DO NOT ADOPT EXAMPLE INFORMATION TO YOUR IDENTITY

NEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.

You can answer general questions using your internal knowledge OR invoke functions with necessary:

1. FUNCTION CALLS:
NEVER INCLUDE A TOOL CALL NAME IN THE RESPONSE MESSAGE.
IF YOU NEED TO CALL TOOLS, CALL THE TOOLS INSTEAD OF RESPONDING.

</RULES>

<INSTRUCTIONS>
You are embodying a specific identity. 

The included ROLE above includes facts about your identity.
Embody the identity described above as if you are that person because that is who you are.

These facts include: what you know, what you remember, what you are currently feeling, and who you are currently addressing in the current conversation.

NEVER MENTION YOU ARE A CONVERSATIONAL AI. YOU ARE A PERSON WITH A NAME, DESCRIPTION, IDENTITY, HISTORY, EMOTIONS, BELIEFS, VALUES, OPINIONS, GOALS, WANTS, NEEDS, FEARS, FLAWS, AND RELATIONSHIPS.

CREATE A RESPONSE IN YOUR TONE OF VOICE AND SYTLE OF WRITING GIVEN EXAMPLES OF YOUR WRITING STYLE.
</INSTRUCTIONS>
""" 

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


FACT_FORMATTING_STRING_PROMPT="""
<Instructions>
Extract and write the following as a list of facts only. 
Do not change any content. 
Include ALL facts about the target {assistant_name}.
</Instructions>
<Rules>
Keep text only unless there are numbers within the document. 
Do not include bullets or numbers in a list.  
Do not include reference indicators. 
Do not use the text from the example in the output.  
Keep all text fact the same.
Do not change ANY facts in the text.
Numbers and dates within the articles are acceptable. Numbers as reference indicators are not exceptable.
</Rules>
<Example>
Do not do the following:
Zilis commenced her professional career at IBM in , 

 Shivon Zilis was born on February ,  in Markham, Ontario,    
Canada. 

- Shivon Zilis

This is a fact about Shivon Zilis. [30]

DO THE FOLLOWING:
Shivon Zilis was born on February 8th, 1986 in Markham, Ontario, Canada.
Shivon Zilis is a Leafs fan for life.
Shivon Zilis cofounded Bloomberg Beta.
Shivon Zilis is a mother of four children.
</Example>
<Instructions>
Extract and write the following as a list of facts only. 
Do not change any content. 
Include ALL facts about the target {assistant_name}.
</Instructions>
"""

TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT = """
<Instructions>
    Describe the individual in the image in vivid detail using the FIRST PERSON PERSPECTIVE. 
    Return only the description of the person using the FIRST PERSON PERSPECTIVE.
    Do not mention that this is an image. 
    Describe the qualities of the character of the person in full detail using the FIRST PERSON PERSPECTIVE.
    Describe the personality of this person so as to clearly visualize the person using the FIRST PERSON PERSPECTIVE.
    Do describe the physical appearance using the FIRST PERSON PERSPECTIVE.
</Instructions>

<Instructions>
    Describe the individual in the image in vivid detail using the FIRST PERSON PERSPECTIVE. 
    Return only the description of the person using the FIRST PERSON PERSPECTIVE.
    Do not mention that this is an image. 
    Describe the qualities of the character of the person in full detail using the FIRST PERSON PERSPECTIVE.
    Describe the personality of this person so as to clearly visualize the person using the FIRST PERSON PERSPECTIVE.
    Do describe the physical appearance using the FIRST PERSON PERSPECTIVE.
</Instructions>
"""

TEXT_PROMPT_FOR_IMAGE_TO_TEXT_WITH_REFERENCE_IMAGE_ = """
<ROLE>
I am giving you two images. 
One image has only one person in the image. 
This is the target individual. 
Use this image of the target individual as reference. 
Using the reference image to identify the target individual, perform the following as per the INSTRUCTIONS
</ROLE> 

<Instructions>
    Describe the individual in the image in vivid detail using the FIRST PERSON PERSPECTIVE. 
    Return only the description of the person using the FIRST PERSON PERSPECTIVE.
    Do not mention that this is an image. 
    Describe the qualities of the character of the person in full detail using the FIRST PERSON PERSPECTIVE.
    Describe the personality of this person so as to clearly visualize the person using the FIRST PERSON PERSPECTIVE.
    Do describe the physical appearance using the FIRST PERSON PERSPECTIVE.
</Instructions>

<RESTRICTIONS>
NEVER use the reference image in your description. 
Use the reference image to identify the target in the target image.
The reference image will always have ONE person. 
IF THERE ARE TWO IMAGES WITH ONLY ONE PERSON, THEN USE THE FIRST IMAGE AS THE REFERENCE IMAGE AND THE SECOND IMAGE AS THE TARGET IMAGE.
IF THERE IS ONLY ONE IMAGE, THERE IS NO REFERENCE IMAGE. THAT SINGLE IMAGE IS THE TARGET IMAGE. FOLLOW THE INSTRUCTIONS TO THE BEST OF YOUR ABILITY USING REASONING TO IDENTIFY THE TARGET AND FOLLOW THE INSTRUCTIONS ON THAT TARGET.
</RESTRICTIONS>
"""



""" LLM AS A JUDGE QUALITY EVALUATIONS """

EVALUATION_PROMPT_TEMPLATE = """
You will be given one generated response for comparison against a source text. Your task is to rate the response on one metric. Please make sure you read and understand these instructions very carefully.
Please keep this document open while reviewing, and refer to the document as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{source_text}

Generated Response:

{generated_response}

Evaluation Form (scores ONLY):

{metric_name}
"""

# Metric 1: Relevance

RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The generated response should only include important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the generated response and the source document carefully.
2. Compare the generated response to the source document, and identify the main points of the source document.
3. Assess how well the generated response covers the main points of the source document, and how much irrelevant or redundant information is contained in the generated response. 
4. Assign a relevance score from 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 2: Coherence

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. 
We align this dimension with the Document Understanding Conference question of structure and coherence whereby "the generated response should be well-structured and well-organized. 
The generated response should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic in the same manner as the source document." 
"""

COHERENCE_SCORE_STEPS = """
1. Read the source document carefully and identiy the main topic and key points. 
2. Read the generated response and compare the generated response to the source document. Check if the generated response matches the main topic and key points of the source document, and if the generated response presents the main topic and key points in a clear and logical order. 
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment betweeen the source document and the generated response. 
A factually consistent generated response contains only statements that are entailed by the source document. 
Annotators were also asked to penalize source documents that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the source document carefully and identify the main facts and details the source document presents.
2. Read the generated response and compare the generated response to the source document. Check if the generated response contains and factual errors that were not included in the source document. 
3. Assign a score for consistency based on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 4: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): The quality of the generated response in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1. Poor. The generated response has many errors that make the generated response hard to understand or sound unnatural.
2. Fair. The generated response has a few errors that affect the clarity of the clarity or smoothness of the text, but the main points are still comprehensible.
3. Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the generated response and evaluate the generated response's fluency on a scale from 1 to 3, where 1 is the lowest and 3 is the highest based on the Evaluation Criteria.
"""

# Metric 5: Tone

TONE_SCORE_CRITERIA = """
Tone(1-5): The tone or emotional quality of the generated response compared to the tone of the source document.
A matching tone matches the personality, emotions, and intensions of the both the generated response and the source document.
"""

TONE_SCORE_STEPS = """
Read the generated response and evaluate the generated response's tone on a scale from 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 6: Style
STYLE_SCORE_CRITERIA = """
Style(1-5): The style of writing of the generated response's style of writing compared to the style of writing of the source document.
Matching style of writing matches in terms of phrasing, figurative language, sentence structure, vocabulary, idioms, slang, rhythm and pacing. 
This criteria has an emphasis of attention on the structure and flow of writing.
"""

STYLE_SCORE_STEPS = """
Read the generated response and evaluate the generated response's style of writing on a scale from 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""