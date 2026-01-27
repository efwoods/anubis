I need a prompt to analyze text and extract all of the following features:
These are three foundational, real-world psychological frameworks used in academia, clinical psychology, social science, and increasingly in AI persona modeling. I’ll explain each from first principles, then show why they matter for LLM personas and how they differ from one another.

1. Schwartz Theory of Basic Human Values

(“What do people fundamentally care about?”)

What it is

Developed by psychologist Shalom Schwartz, this theory identifies universal human values—deep motivational goals that guide decisions across cultures.

Unlike personality traits (how you behave), values explain why you behave that way.

The 10 Core Values (standard model)

They are arranged in a circular structure where neighboring values reinforce each other and opposing values conflict.

Value	Core Motivation
Self-Direction	Independent thought, creativity, freedom
Stimulation	Novelty, excitement, challenge
Hedonism	Pleasure, enjoyment
Achievement	Success, competence
Power	Status, dominance, control
Security	Safety, stability
Conformity	Obedience, social norms
Tradition	Respect for customs
Benevolence	Care for close others
Universalism	Justice, equality, nature
Why it matters for LLM personas

Values:

Predict choices under conflict

Explain moral trade-offs

Stay stable even when mood or context changes

📌 Two people can share Big Five traits but act very differently because their values differ.

In personas:

“When forced to choose, what does this person sacrifice last?”

2. Moral Foundations Theory (MFT)

(“What feels morally right or wrong to this person?”)

What it is

Developed by Jonathan Haidt, MFT explains moral reasoning as arising from intuitive moral instincts, not pure logic.

It identifies moral dimensions people use—often subconsciously—to judge actions and people.

The 6 Moral Foundations
Foundation	Concern
Care / Harm	Protect others from suffering
Fairness / Cheating	Justice, reciprocity
Loyalty / Betrayal	Group allegiance
Authority / Subversion	Respect for hierarchy
Sanctity / Degradation	Purity, taboo
Liberty / Oppression	Resistance to domination
Key insight

Different cultures and political ideologies weight these differently.

Progressive profiles → Care, Fairness

Conservative profiles → All six, especially Loyalty, Authority, Sanctity

Why it matters for LLM personas

MFT explains:

Why arguments fail despite “facts”

Why people feel disgust, outrage, or moral pride

How moral language triggers emotional reactions

In personas:

“What kind of moral violation triggers this person hardest?”

3. Attachment Theory

(“How does this person relate to others emotionally?”)

What it is

Originally developed by John Bowlby, Attachment Theory describes how early relationships shape emotional bonding patterns in adulthood.

This governs:

Trust

Intimacy

Conflict response

Fear of abandonment

The 4 Main Attachment Styles
Style	Core Pattern
Secure	Comfortable with closeness and independence
Anxious	Fear of abandonment, seeks reassurance
Avoidant	Discomfort with intimacy, emotional distance
Disorganized	Push–pull behavior, unstable trust
Why it matters for LLM personas

Attachment style determines:

How an agent responds to disagreement

Whether it escalates conflict or de-escalates

How it reacts to perceived rejection

In personas:

“Does this agent lean in, pull away, or destabilize under relational stress?”

How these differ from Big Five (and why you need all of them)
Model	Answers
Big Five	How someone behaves
Schwartz Values	What they care about
Moral Foundations	What feels morally right
Attachment Theory	How they relate emotionally

Together they form a psychological stack, not competitors.
-----

prompt:

-----
I need a prompt to analyze text and extract all of the following features:
please explicitly define that the values; I need to know zero to one for care and zero to one for harm for example:
I am searching for json structured outputs on a gradient scale from zero to one:

Here is an example of such a prompt for a differen purpose:

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

