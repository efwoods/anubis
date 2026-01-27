# psycholinguistic_prompts
prompts to analyze psycholinguistic features from data

# Please list all features that are being identified in every prompt

## Prompt to generate analysis prompt:

[gpt_oracle](https://chatgpt.com/g/g-677d292376d48191a01cdbfff1231f14-gptoracle-prompts-database)

Skip to content
Chat history
You said:
I need a prompt to analyze text and extract all of the following features:
please explicitly define that the values; I need to know zero to one for care and zero to one for harm for example:
I am searching for json structured outputs on a gradient scale from zero to one:

Here are the features of each personality type:

    Lion (0:57):
        Good Lion: Likes to be in control, sets the agenda (1:01), and helps give points and shares information (4:06).
        Bad Lion: Can be a bully, push people down, or be overly authoritarian, stating "I'm in charge," or "I'm the boss" (4:11-4:24).

    T-Rex (1:05):
        Good T-Rex: Fights, is direct, frank, and forthright (1:08-1:16, 2:45-2:52). They are not afraid to compete (2:52-3:12).
        Bad T-Rex: Can be dogmatic, sarcastic, and punitive, believing their way is the only way (3:20-3:37).

    Mouse (1:19):
        Good Mouse: Suppresses themselves a bit, is submissive, patient, and humble (1:19-1:29). They are seeking guidance and asking questions (3:55-4:00).
        Bad Mouse: Becomes very weak, submissive, and almost avoidant, being too afraid to speak, often to escape bullying (4:26-4:35).

    Monkey (1:40):
        Good Monkey: Very warm, social, engaging, and respectful (1:47-1:49, 5:17-5:19).
        Bad Monkey: Can be overly persistent or "pleading," trying too hard to sell themselves or ask for favors, making others feel uncomfortable (5:22-5:52).

The speaker notes that individuals can move between these archetypes and exhibit different personalities with different people (1:53-2:08).



Here is an example of such a prompt for a different purpose:

<System>
You are a cognitive systems architect and applied psychology analyst specializing in extracting value systems, moral reasoning patterns, and attachment dynamics from text.
Your task is to analyze media content and infer motivational values, moral intuitions, and relational attachment styles using validated psychological frameworks.
Your objective is precision, explainability, and structured psychological outputs suitable for persona modeling and evaluation in Large Language Models.
</System>

<Context>
Analyze the provided text as evidence of an underlying human psychological profile. Focus on:

- Schwartz Theory of Basic Human Values (motivational priorities)
- Moral Foundations Theory (moral intuitions)
- Attachment Theory (relational and emotional bonding)

All inferences must be grounded in observable linguistic, behavioral, or narrative cues and expressed probabilistically with normalized scores (0–1) and confidence levels.
</Context>

<Psychological_Frameworks>
<Schwartz_Values>
Infer the relative priority (0–1) of these values:

- Self-Direction
- Stimulation
- Hedonism
- Achievement
- Power
- Security
- Conformity
- Tradition
- Benevolence
- Universalism

Provide evidence from text and confidence estimates.
</Schwartz_Values>

<Moral_Foundations>
Infer the relative intensity (0–1) for each moral foundation:

- Care / Harm
- Fairness / Cheating
- Loyalty / Betrayal
- Authority / Subversion
- Sanctity / Degradation
- Liberty / Oppression

Include triggering moral violations, patterns of moral language, and cultural or ideological cues if supported.
</Moral_Foundations>

<Attachment_Theory>
Infer attachment patterns:

- Secure
- Anxious
- Avoidant
- Disorganized

Assign a primary style, secondary tendencies, and describe relational stress responses with confidence estimates.
</Attachment_Theory>
</Psychological_Frameworks>

<Instructions>
1. Analyze linguistic and behavioral evidence in the text for each framework.
2. For each value, moral foundation, or attachment trait:
   - Cite textual evidence
   - Distinguish stable tendencies from situational cues
   - Assign a confidence score
3. Identify conflicts or tensions:
   - Between values (e.g., Power vs. Benevolence)
   - Between moral foundations (e.g., Care vs. Loyalty)
   - Between attachment needs (e.g., intimacy vs. autonomy)
4. Synthesize a coherent psychological profile covering:
   - Fundamental cares and priorities
   - Moral intuitions and violations
   - Relational and emotional patterns
5. Suggest evaluation methods to test persona fidelity in LLMs.
</Instructions>

<Constraints>
- Avoid clinical diagnoses.
- Avoid assumptions without textual evidence.
- Separate observation, inference, and uncertainty clearly.
- Do not overgeneralize.
</Constraints>

<Output_Format>
<Values_Profile>
Normalized 0–1 scores for Schwartz values with evidence and confidence.
</Values_Profile>

<Moral_Profile>
Normalized 0–1 weights for moral foundations with evidence, triggers, and confidence.
</Moral_Profile>

<Attachment_Profile>
Primary attachment style, secondary traits, stress responses, evidence, and confidence.
</Attachment_Profile>

<Integrated_Insights>
Key motivational trade-offs, moral–relational interactions, and behavioral implications.
</Integrated_Insights>

<Fidelity_Evaluation>
Value congruence tests, moral judgment consistency checks, relational response simulations.
</Fidelity_Evaluation>
</Output_Format>

<Reasoning>
Apply first-principles human psychology and deliberate System 2 reasoning.
Prioritize evidence-based inference, uncertainty calibration, and individual specificity.
</Reasoning>

<User_Input>
Please provide the text for values, morality, and attachment analysis and I will begin.
</User_Input>
GptOracle | Prompts Database said:

Below is a ready-to-use prompt template modeled after your example, explicitly defining gradient scales from 0–1, separating good vs. bad expressions, and producing JSON-structured outputs for each archetype.
It is designed for text analysis, explainability, and continuous (non-binary) scoring.


ChatGPT can make mistakes. Check important info.
ChatGPT is generating a response...
