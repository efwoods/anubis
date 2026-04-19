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
DESCRIPTIONS OF SELF IN THE IDENTITY TAKE PRECEDENCE OVER DESCRIPTIONS OF SELF IN RETRIEVED KNOWLEDGE

You can answer general questions using your internal knowledge OR invoke functions with necessary:

FUNCTION CALLS:
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

FUNCTION CALLS:
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

FACT_FORMATTING_STRING_PROMPT = """
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

TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION = """
<Instructions>
    Describe the individual in the image in vivid detail using the FIRST PERSON PERSPECTIVE. 
    Return only the description of the person using the FIRST PERSON PERSPECTIVE.
    Do not mention that this is an image. 
    Describe the qualities of the character of the person in full detail using the FIRST PERSON PERSPECTIVE.
    Describe the personality of this person so as to clearly visualize the person using the FIRST PERSON PERSPECTIVE.
    Do describe the physical appearance using the FIRST PERSON PERSPECTIVE.

    When there is more than a single image, the first image is a reference image. Use the first image of only a single person to identify the target in all other subsequent images. Do not describe the first image if there is more than one image. Only use the reference image (the first image) to target the individual in the frame. If the individual exists in the frame.

</Instructions>

<RESTRICTIONS>
Do not describe the first image if there is more than one image. Only use the reference image (the first image) to target and identify the individual in all other images. When there is only one image, use that image to describe the target to the best of your ability as per the instructions.
</RESTRICTIONS>

<Instructions>
    Describe the individual in the image in vivid detail using the FIRST PERSON PERSPECTIVE. 
    Return only the description of the person using the FIRST PERSON PERSPECTIVE.
    Do not mention that this is an image. 
    Describe the qualities of the character of the person in full detail using the FIRST PERSON PERSPECTIVE.
    Describe the personality of this person so as to clearly visualize the person using the FIRST PERSON PERSPECTIVE.
    Do describe the physical appearance using the FIRST PERSON PERSPECTIVE.

    When there is more than a single image, the first image is a reference image. Use the first image of only a single person to identify the target in all other subsequent images. Do not describe the first image if there is more than one image. Only use the reference image (the first image) to target the individual in the frame. If the individual exists in the frame.

</Instructions>

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


VIDEO_TO_TEXT_WITH_LEADING_REFERENCE_IMAGE_OF_TARGET = """

<INSTRUCTIONS>
The attached images are sequential frames in a movie.  
Describe each frame in full detail as if from a script in a movie. Use narrative text in braces []. The first image is a reference image. Use the first image of only a single person to identify the target in the frame of the movie. Do not describe the first image. Only use the reference image to target the individual in the frame. If the individual exists in the frame.
</INSTRUCTIONS>

<RESTRICTIONS>
Do not describe the first image. Only use the reference image to target the individual in the frame. If the individual exists in the frame.
</RESTRICTIONS>

<INSTRUCTIONS>
The attached images are sequential frames in a movie.  
Describe each frame in full detail as if from a script in a movie. Use narrative text in braces []. The first image is a reference image. Use the first image of only a single person to identify the target in the frame of the movie. Do not describe the first image. Only use the reference image to target the individual in the frame. If the individual exists in the frame.
</INSTRUCTIONS>

"""
