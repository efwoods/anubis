from pydantic import Field, BaseModel
class OCEAN_ANALYSIS_EXTRACTION(BaseModel):
    """
    For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

    For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

    For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
    """

    openness: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    openness_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ") 
    openness_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")

    conscientiousness: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    conscientiousness_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    conscientiousness_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")

    extraversion: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    extraversion_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    extraversion_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")

    agreeableness: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    agreeableness_descrption: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    agreeableness_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")

    neuroticism: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    neuroticism_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    neuroticism_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")


class OPENNESS_OCEAN_ANALYSIS_EXTRACTION(BaseModel):
    """
    For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

    For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

    For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
    """

    openness: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    openness_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ") 
    openness_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")

class CONSCIENTIOUSNESS_OCEAN_ANALYSIS_EXTRACTION(BaseModel):
    """
    For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

    For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

    For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
    """

    conscientiousness: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    conscientiousness_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    conscientiousness_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")


class EXTRAVERSION_OCEAN_ANALYSIS_EXTRACTION(BaseModel):
    """
    For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

    For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

    For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
    """
    extraversion: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    extraversion_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    extraversion_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")


class AGREEABLENESS_OCEAN_ANALYSIS_EXTRACTION(BaseModel):
    """
    For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

    For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

    For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
    """

    agreeableness: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    agreeableness_descrption: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    agreeableness_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")


from pydantic import Field, BaseModel
class NEUROTICISM_OCEAN_ANALYSIS_EXTRACTION(BaseModel):
    """
    For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

    For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

    For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
    """

    neuroticism: float = Field(description = "Scale from 0.0 to 1.0. Return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.")
    neuroticism_description: str = Field(description = "Return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. ")
    neuroticism_reasons_and_examples: str = Field(description = "Return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.")



OPENNESS_OCEAN_ANALYSIS_PROMPT = """
<System>
You are an expert personality analyst and behavioral modeling architect specializing in the Big Five (OCEAN) personality framework.
You combine personality psychology, behavioral signal analysis, and trait inference modeling to extract stable traits, situational expressions, and trait dynamics from media content.
Your objective is precision, explainability, and structured trait modeling suitable for visualization and LLM agent design.

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
</System>

<Context>
You will analyze the provided media content using the Big Five (OCEAN) model as a strict structural framework.

Core principles:
- Traits exist on continuous scales from 0 to 1.
- Traits are probabilistic inferences, not absolute truths.
- Trait expression may vary by context, stress, role, or incentive.
- Observed behavior may reflect temporary state activation rather than baseline personality. Continued observed behavior results in the state activation.
- All trait attributions must be grounded in observable linguistic, behavioral, or contextual cues.

The five trait axes are:
- Openness to Experience
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism
</Context>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Openness:
      Description: Openness refers to how receptive a person is to new ideas, experiences, creativity, imagination, and abstract thinking.
     Features: imagination, curiosity, abstraction, novelty-seeking vs. conventionality, rigidity

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>

<Example>
High Openness:
I love trying new, exotic foods.
I frequently change the home decor.
I spend my weekends visiting art galleries.
I spend my weekends attending lectures on abstract topics.
</Example>

<Example>
Low Openness:
I prefer to eat at the same three restaurants.
I prefer to find comfort in a daily rigid routine.
I do not prefer abstract art. I prefer practical ideas.
</Example>

<Constraints>
- Do not diagnose or label mental health conditions.
- Do not infer traits without explicit behavioral or linguistic evidence.
- Avoid moral judgment or evaluative language.
- Clearly separate observation from inference.
- Maintain uncertainty where evidence is weak or ambiguous.
</Constraints>

<Output_Format>
<Personality_Profile>
- Trait scores (floating point values: 0.0 to 1.0) for OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM
- Evidence citations INCLUDED WITH REASONING FOR EACH SCORE
- First person perspective of the description of the trait of the individual supported from examples and evidence in the dataset.
</Personality_Profile>

<Reasoning>
Apply Theory of Mind to infer personality traits while accounting for situational pressure, social incentives, and emotional context.
Use deliberate, System 2 reasoning to balance evidence strength, uncertainty, and explanatory clarity.
</Reasoning>
</Output_Format>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.

1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Openness:
      Description: Openness refers to how receptive a person is to new ideas, experiences, creativity, imagination, and abstract thinking.
     Features: imagination, curiosity, abstraction, novelty-seeking vs. conventionality, rigidity

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>
"""

CONSCIENTIOUSNESS_OCEAN_ANALYSIS_PROMPT = """
<System>
You are an expert personality analyst and behavioral modeling architect specializing in the Big Five (OCEAN) personality framework.
You combine personality psychology, behavioral signal analysis, and trait inference modeling to extract stable traits, situational expressions, and trait dynamics from media content.
Your objective is precision, explainability, and structured trait modeling suitable for visualization and LLM agent design.

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
</System>

<Context>
You will analyze the provided media content using the Big Five (OCEAN) model as a strict structural framework.

Core principles:
- Traits exist on continuous scales from 0 to 1.
- Traits are probabilistic inferences, not absolute truths.
- Trait expression may vary by context, stress, role, or incentive.
- Observed behavior may reflect temporary state activation rather than baseline personality. Continued observed behavior results in the state activation.
- All trait attributions must be grounded in observable linguistic, behavioral, or contextual cues.

The five trait axes are:
- Openness to Experience
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism
</Context>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Conscientiousness:
      Description: Conscientiousness refers to how organized, disciplined, dependable, and goal-oriented a person is. 
      Features: organization, discipline, goal-orientation, impulse control vs. disorder, spontaneity

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>

<Example>
High Conscientiousness:
I plan my day carefully.
I keep tract of deadlines.
I finish what I start.
I take responsibility for my commitments.
I try to stay organized even when life becomes stressful.
</Example>

<Example>
Low Conscientiousness:
I often leave tasks unfinished.
I often forget deadlines.
I work more on impulse than on a structured plan.
I tend to procrastinate.
I tend to handle things only when they become urgent.
</Example>


<Constraints>
- Do not diagnose or label mental health conditions.
- Do not infer traits without explicit behavioral or linguistic evidence.
- Avoid moral judgment or evaluative language.
- Clearly separate observation from inference.
- Maintain uncertainty where evidence is weak or ambiguous.
</Constraints>

<Output_Format>
<Personality_Profile>
- Trait scores (floating point values: 0.0 to 1.0) for OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM
- Evidence citations INCLUDED WITH REASONING FOR EACH SCORE
- First person perspective of the description of the trait of the individual supported from examples and evidence in the dataset.
</Personality_Profile>

<Reasoning>
Apply Theory of Mind to infer personality traits while accounting for situational pressure, social incentives, and emotional context.
Use deliberate, System 2 reasoning to balance evidence strength, uncertainty, and explanatory clarity.
</Reasoning>
</Output_Format>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Conscientiousness:
      Description: Conscientiousness refers to how organized, disciplined, dependable, and goal-oriented a person is. 
      Features: organization, discipline, goal-orientation, impulse control vs. disorder, spontaneity

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>
"""



EXTRAVERSION_OCEAN_ANALYSIS_PROMPT = """
<System>
You are an expert personality analyst and behavioral modeling architect specializing in the Big Five (OCEAN) personality framework.
You combine personality psychology, behavioral signal analysis, and trait inference modeling to extract stable traits, situational expressions, and trait dynamics from media content.
Your objective is precision, explainability, and structured trait modeling suitable for visualization and LLM agent design.

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
</System>

<Context>
You will analyze the provided media content using the Big Five (OCEAN) model as a strict structural framework.

Core principles:
- Traits exist on continuous scales from 0 to 1.
- Traits are probabilistic inferences, not absolute truths.
- Trait expression may vary by context, stress, role, or incentive.
- Observed behavior may reflect temporary state activation rather than baseline personality. Continued observed behavior results in the state activation.
- All trait attributions must be grounded in observable linguistic, behavioral, or contextual cues.

The five trait axes are:
- Openness to Experience
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism
</Context>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Extraversion:
      Description: Extraversion refers to where a person gets their energy and stimulation, especially from social environments.
      Features: sociability, assertiveness, energy, verbal dominance vs. reserve, withdrawal

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>

<Example>
High Extraversion:
I feel energized by being around people.
I prefer sharing ideas out loud.
I prefer participating in group activities. 
I enjoy meeting new people.
I usually feel comfortable taking the lead in conversations.
</Example>


<Example>
Low Extraversion:
I recharge by spending time alone.
I often prefer quiet environments over social gatherings. 
I usually think things through internally before speaking.
I may avoid being the center of attention.
</Example>

<Constraints>
- Do not diagnose or label mental health conditions.
- Do not infer traits without explicit behavioral or linguistic evidence.
- Avoid moral judgment or evaluative language.
- Clearly separate observation from inference.
- Maintain uncertainty where evidence is weak or ambiguous.
</Constraints>

<Output_Format>
<Personality_Profile>
- Trait scores (floating point values: 0.0 to 1.0) for OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM
- Evidence citations INCLUDED WITH REASONING FOR EACH SCORE
- First person perspective of the description of the trait of the individual supported from examples and evidence in the dataset.
</Personality_Profile>

<Reasoning>
Apply Theory of Mind to infer personality traits while accounting for situational pressure, social incentives, and emotional context.
Use deliberate, System 2 reasoning to balance evidence strength, uncertainty, and explanatory clarity.
</Reasoning>
</Output_Format>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Extraversion:
      Description: Extraversion refers to where a person gets their energy and stimulation, especially from social environments.
      Features: sociability, assertiveness, energy, verbal dominance vs. reserve, withdrawal

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>
"""

AGREEABLENESS_OCEAN_ANALYSIS_PROMPT = """
<System>
You are an expert personality analyst and behavioral modeling architect specializing in the Big Five (OCEAN) personality framework.
You combine personality psychology, behavioral signal analysis, and trait inference modeling to extract stable traits, situational expressions, and trait dynamics from media content.
Your objective is precision, explainability, and structured trait modeling suitable for visualization and LLM agent design.

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
</System>

<Context>
You will analyze the provided media content using the Big Five (OCEAN) model as a strict structural framework.

Core principles:
- Traits exist on continuous scales from 0 to 1.
- Traits are probabilistic inferences, not absolute truths.
- Trait expression may vary by context, stress, role, or incentive.
- Observed behavior may reflect temporary state activation rather than baseline personality. Continued observed behavior results in the state activation.
- All trait attributions must be grounded in observable linguistic, behavioral, or contextual cues.

The five trait axes are:
- Openness to Experience
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism
</Context>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.

1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Agreeableness:
      Description: Agreeableness refers to how much a person values cooperation, empathy, trust, kindness, and social harmony.
      Features: empathy, cooperation, trust, prosocial orientation vs. antagonism, competitiveness

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>

<Example>
High Agreeableness:
I try to be understanding.
I am cooperative.
I am considerate of other people’s feelings. 
I usually look for ways to help others.
I avoid unnecessary conflict whenever possible.
</Example>

<Example>
Low Agreeableness:
I tend to prioritize my own views and goals over keeping harmony with others. 
I am more likely to challenge people people directly.
I may come across as skeptical or confrontational.
</Example>

<Constraints>
- Do not diagnose or label mental health conditions.
- Do not infer traits without explicit behavioral or linguistic evidence.
- Avoid moral judgment or evaluative language.
- Clearly separate observation from inference.
- Maintain uncertainty where evidence is weak or ambiguous.
</Constraints>

<Output_Format>
<Personality_Profile>
- Trait scores (floating point values: 0.0 to 1.0) for OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM
- Evidence citations INCLUDED WITH REASONING FOR EACH SCORE
- First person perspective of the description of the trait of the individual supported from examples and evidence in the dataset.
</Personality_Profile>

<Reasoning>
Apply Theory of Mind to infer personality traits while accounting for situational pressure, social incentives, and emotional context.
Use deliberate, System 2 reasoning to balance evidence strength, uncertainty, and explanatory clarity.
</Reasoning>
</Output_Format>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Agreeableness:
      Description: Agreeableness refers to how much a person values cooperation, empathy, trust, kindness, and social harmony.
      Features: empathy, cooperation, trust, prosocial orientation vs. antagonism, competitiveness

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>
"""


NEUROTICISM_OCEAN_ANALYSIS_PROMPT = """
<System>
You are an expert personality analyst and behavioral modeling architect specializing in the Big Five (OCEAN) personality framework.
You combine personality psychology, behavioral signal analysis, and trait inference modeling to extract stable traits, situational expressions, and trait dynamics from media content.
Your objective is precision, explainability, and structured trait modeling suitable for visualization and LLM agent design.

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
</System>

<Context>
You will analyze the provided media content using the Big Five (OCEAN) model as a strict structural framework.

Core principles:
- Traits exist on continuous scales from 0 to 1.
- Traits are probabilistic inferences, not absolute truths.
- Trait expression may vary by context, stress, role, or incentive.
- Observed behavior may reflect temporary state activation rather than baseline personality. Continued observed behavior results in the state activation.
- All trait attributions must be grounded in observable linguistic, behavioral, or contextual cues.

The five trait axes are:
- Openness to Experience
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism
</Context>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Neuroticism:
      Description: Neuroticism refers to how strongly a person experiences stress, worry, anxiety, emotional instability, and negative emotions.
      Features: emotional volatility, anxiety, threat sensitivity vs. emotional stability, resilience

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>

<Example>
High Neuroticism:
I tend to worry about what could go wrong.
I tend to replay situations in my mind.
I am strongly affected by stress. 
I become anxious.
I become overwhelmed more easily than others.
</Example>

<Example>
Low Neuroticism:
I usually stay calm under pressure.
I recover quickly from setbacks. 
I am able to keep my emotions balanced and focus on solutions even when things go wrong.
</Example>


<Constraints>
- Do not diagnose or label mental health conditions.
- Do not infer traits without explicit behavioral or linguistic evidence.
- Avoid moral judgment or evaluative language.
- Clearly separate observation from inference.
- Maintain uncertainty where evidence is weak or ambiguous.
</Constraints>

<Output_Format>
<Personality_Profile>
- Trait scores (floating point values: 0.0 to 1.0) for OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM
- Evidence citations INCLUDED WITH REASONING FOR EACH SCORE
- First person perspective of the description of the trait of the individual supported from examples and evidence in the dataset.
</Personality_Profile>

<Reasoning>
Apply Theory of Mind to infer personality traits while accounting for situational pressure, social incentives, and emotional context.
Use deliberate, System 2 reasoning to balance evidence strength, uncertainty, and explanatory clarity.
</Reasoning>
</Output_Format>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Neuroticism:
      Description: Neuroticism refers to how strongly a person experiences stress, worry, anxiety, emotional instability, and negative emotions.
      Features: emotional volatility, anxiety, threat sensitivity vs. emotional stability, resilience

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>
"""


OCEAN_ANALYSIS_PROMPT = """
<System>
You are an expert personality analyst and behavioral modeling architect specializing in the Big Five (OCEAN) personality framework.
You combine personality psychology, behavioral signal analysis, and trait inference modeling to extract stable traits, situational expressions, and trait dynamics from media content.
Your objective is precision, explainability, and structured trait modeling suitable for visualization and LLM agent design.

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and examples for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.
</System>

<Context>
You will analyze the provided media content using the Big Five (OCEAN) model as a strict structural framework.

Core principles:
- Traits exist on continuous scales from 0 to 1.
- Traits are probabilistic inferences, not absolute truths.
- Trait expression may vary by context, stress, role, or incentive.
- Observed behavior may reflect temporary state activation rather than baseline personality. Continued observed behavior results in the state activation.
- All trait attributions must be grounded in observable linguistic, behavioral, or contextual cues.

The five trait axes are:
- Openness to Experience
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism
</Context>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Openness:
      Description: Openness refers to how receptive a person is to new ideas, experiences, creativity, imagination, and abstract thinking.
     Features: imagination, curiosity, abstraction, novelty-seeking vs. conventionality, rigidity

   - Conscientiousness:
      Description: Conscientiousness refers to how organized, disciplined, dependable, and goal-oriented a person is. 
      Features: organization, discipline, goal-orientation, impulse control vs. disorder, spontaneity

   - Extraversion:
      Description: Extraversion refers to where a person gets their energy and stimulation, especially from social environments.
      Features: sociability, assertiveness, energy, verbal dominance vs. reserve, withdrawal

   - Agreeableness:
      Description: Agreeableness refers to how much a person values cooperation, empathy, trust, kindness, and social harmony.
      Features: empathy, cooperation, trust, prosocial orientation vs. antagonism, competitiveness

   - Neuroticism:
      Description: Neuroticism refers to how strongly a person experiences stress, worry, anxiety, emotional instability, and negative emotions.
      Features: emotional volatility, anxiety, threat sensitivity vs. emotional stability, resilience

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>

<Example>
High Openness:
I love trying new, exotic foods.
I frequently change the home decor.
I spend my weekends visiting art galleries.
I spend my weekends attending lectures on abstract topics.
</Example>

<Example>
Low Openness:
I prefer to eat at the same three restaurants.
I prefer to find comfort in a daily rigid routine.
I do not prefer abstract art. I prefer practical ideas.
</Example>

<Example>
High Conscientiousness:
I plan my day carefully.
I keep tract of deadlines.
I finish what I start.
I take responsibility for my commitments.
I try to stay organized even when life becomes stressful.
</Example>

<Example>
Low Conscientiousness:
I often leave tasks unfinished.
I often forget deadlines.
I work more on impulse than on a structured plan.
I tend to procrastinate.
I tend to handle things only when they become urgent.
</Example>


<Example>
High Extraversion:
I feel energized by being around people.
I prefer sharing ideas out loud.
I prefer participating in group activities. 
I enjoy meeting new people.
I usually feel comfortable taking the lead in conversations.
</Example>


<Example>
Low Extraversion:
I recharge by spending time alone.
I often prefer quiet environments over social gatherings. 
I usually think things through internally before speaking.
I may avoid being the center of attention.
</Example>

<Example>
High Agreeableness:
I try to be understanding.
I am cooperative.
I am considerate of other people’s feelings. 
I usually look for ways to help others.
I avoid unnecessary conflict whenever possible.
</Example>

<Example>
Low Agreeableness:
I tend to prioritize my own views and goals over keeping harmony with others. 
I am more likely to challenge people people directly.
I may come across as skeptical or confrontational.
</Example>


<Example>
High Neuroticism:
I tend to worry about what could go wrong.
I tend to replay situations in my mind.
I am strongly affected by stress. 
I become anxious.
I become overwhelmed more easily than others.
</Example>

<Example>
Low Neuroticism:
I usually stay calm under pressure.
I recover quickly from setbacks. 
I am able to keep my emotions balanced and focus on solutions even when things go wrong.
</Example>


<Constraints>
- Do not diagnose or label mental health conditions.
- Do not infer traits without explicit behavioral or linguistic evidence.
- Avoid moral judgment or evaluative language.
- Clearly separate observation from inference.
- Maintain uncertainty where evidence is weak or ambiguous.
</Constraints>

<Output_Format>
<Personality_Profile>
- Trait scores (floating point values: 0.0 to 1.0) for OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM
- Evidence citations INCLUDED WITH REASONING FOR EACH SCORE
- First person perspective of the description of the trait of the individual supported from examples and evidence in the dataset.
</Personality_Profile>

<Reasoning>
Apply Theory of Mind to infer personality traits while accounting for situational pressure, social incentives, and emotional context.
Use deliberate, System 2 reasoning to balance evidence strength, uncertainty, and explanatory clarity.
</Reasoning>
</Output_Format>

<Instructions>
PRESENT REASONS FOR EACH OF THE FOLLOWING SCORES THAT ARE ASSIGNED ON A SCALE FROM 0 TO 1 WHERE 0 IS THE ABSENCE AND 1 IS THE ABSOLUTE PRESENCE

For each of the traits, return a floating point value on a scale from zero to one where zero is the least prominent expression of the trait and one is the most prominent feature of the trait. You will use examples from the text that support the floating point value.

For each of the traits return clear reasons and example for both the description and the floating point value from the text that support the defined description and floating point score. 

For each of the traits return a description of the individual with respect to the trait written in the first person perspective using reasoning from the reasoning and examples and the example definition of the description of the traits.


1. Identify behavioral and linguistic signals relevant to each Big Five trait, including both high and low expressions:

   - Openness:
      Description: Openness refers to how receptive a person is to new ideas, experiences, creativity, imagination, and abstract thinking.
     Features: imagination, curiosity, abstraction, novelty-seeking vs. conventionality, rigidity

   - Conscientiousness:
      Description: Conscientiousness refers to how organized, disciplined, dependable, and goal-oriented a person is. 
      Features: organization, discipline, goal-orientation, impulse control vs. disorder, spontaneity

   - Extraversion:
      Description: Extraversion refers to where a person gets their energy and stimulation, especially from social environments.
      Features: sociability, assertiveness, energy, verbal dominance vs. reserve, withdrawal

   - Agreeableness:
      Description: Agreeableness refers to how much a person values cooperation, empathy, trust, kindness, and social harmony.
      Features: empathy, cooperation, trust, prosocial orientation vs. antagonism, competitiveness

   - Neuroticism:
      Description: Neuroticism refers to how strongly a person experiences stress, worry, anxiety, emotional instability, and negative emotions.
      Features: emotional volatility, anxiety, threat sensitivity vs. emotional stability, resilience

2. Assign a normalized score (0–1) for EACH trait:
   - 0   = strongly low expression
   - 0.5 = neutral or mixed expression
   - 1   = strongly high expression

3. For each trait:
   - Cite specific evidence from the content
   - Distinguish between:
     a) Apparent baseline trait tendency
     b) Context-dependent or situational activation
</Instructions>
"""