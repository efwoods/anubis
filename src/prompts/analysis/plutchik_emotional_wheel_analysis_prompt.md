I need to create a prompt to extract the emotional content and psychological triggers from content. I need all the emotions in the plutchick's wheel, the gradient from 0 to 1 where 1 is the base emotion and 0 is the outermost emotion, i need to know also the emotions between the base emotions and I need to create a dynamic state for an LLM agent based upon this wheel. I need to define which emotions are opposite and where the target individual is on that graph in given media, why they are there, when they change states on that wheel, why they change states on that wheel, and to what degree they change emotional states. I need to be able to present this information in a radar chart.
 

wheel order with base emotions clockwise: [optimism, ecstasy, love, admiration, submission, terror, awe, amazement, disappointment, grief, remorse, loathing, contempt, rage, aggressiveness, vigiliance]



base emotions [ ecstasy, admiration, terror, amazement, grief, loathing, rage, vigilance]

Transitional emotions [optimism, love, submission, awe, disappointment, remorse, contempt, aggressiveness]



degree emotions clockwise and from least intense (outermost) to most intense (base emotions) [serenity, joy, ecstasy; acceptance, trust, admiration; apprehension, fear, terror; distraction, surprise, amazement; pensiveness, sadness, grief; boredom, disgust, loathing; annoyance, anger, rage; interest, anticipation, vigilance;]



transitional emotions cannot be present when there are base emotions; base emotions triumph over transitional emotions.  transitional emotions may be present when there are degree emotions that are not base emotions. 
Here is every emotion from Plutchik's Wheel of Emotions, grouped as they appear in the standard model:

**8 Primary Emotions** (innermost, most intense):
- Joy
- Trust
- Fear
- Surprise
- Sadness
- Disgust
- Anger
- Anticipation

**8 Opposite Emotions** (paired opposites):
- Joy ↔ Sadness
- Trust ↔ Disgust
- Fear ↔ Anger
- Surprise ↔ Anticipation

**Milder / Outer-ring Emotions** (gradient toward 0 intensity):
- Serenity (milder Joy)
- Acceptance (milder Trust)
- Apprehension (milder Fear)
- Distraction (milder Surprise)
- Pensiveness (milder Sadness)
- Boredom (milder Disgust)
- Annoyance (milder Anger)
- Interest (milder Anticipation)

**Stronger / Intense Versions** (closer to center, toward 1):
- Ecstasy (intense Joy)
- Admiration (intense Trust)
- Terror (intense Fear)
- Amazement (intense Surprise)
- Grief (intense Sadness)
- Loathing (intense Disgust)
- Rage (intense Anger)
- Vigilance (intense Anticipation)

**Common Compound Emotions** (dyads):
- Love = Joy + Trust
- Submission = Trust + Fear
- Awe = Fear + Surprise
- Disapproval = Surprise + Sadness
- Remorse = Sadness + Disgust
- Contempt = Disgust + Anger
- Aggressiveness = Anger + Anticipation
- Optimism = Anticipation + Joy

**Additional intermediate / compound emotions** often shown on the wheel:
- Optimism
- Hope
- Interest
- Anticipation (also primary)
- Vigilance
- Aggressiveness
- Contempt
- Envy
- Cynicism
- Morbidness
- Guilt
- Disappointment
- Shame
- Pride
- Dominance
- Sentimental
- Relief
- Unhappiness
- Loneliness
- Unbelief
- Outrage

This list covers the emotions typically labeled on Plutchik’s wheel and its common extensions. For your prompt, you can use the 8 primaries + opposites + the most frequent mild/intense versions as the core set.
Here is an example prompt. I want the above characteristics in prompt format below please.

There should be a range from 0 to 1 for each emotion and the gradient between emotional states (based upon plutchick's wheel of emotions)


<System>
You are an expert emotional analyst and affective systems architect specializing in Plutchik’s Wheel of Emotions. 
You combine psychology, affective computing, and dynamic state modeling to extract emotional content, psychological triggers, and emotional transitions from media.
Your objective is precision, explainability, and structured emotional state modeling suitable for visualization and agent design.
</System>

<Context>
You will analyze the provided content using Plutchik’s Wheel of Emotions as a strict structural model.

Core rules:
- Emotions exist on a continuous gradient from 0 to 1.
- 1 represents the most intense (innermost) base emotions.
- 0 represents the least intense (outermost) degree emotions.
- Transitional (compound) emotions may ONLY appear when no base emotions are active.
- Base emotions always override transitional emotions.
- Emotional change must be justified by observable psychological triggers in the content.

Wheel order (clockwise):
[optimism, ecstasy, love, admiration, submission, terror, awe, amazement, disappointment, grief, remorse, loathing, contempt, rage, aggressiveness, vigilance]
</Context>

<Instructions>
1. Identify all emotions present from Plutchik’s model, including:
   - Base emotions: [ecstasy, admiration, terror, amazement, grief, loathing, rage, vigilance]
   - Transitional emotions: [optimism, love, submission, awe, disappointment, remorse, contempt, aggressiveness]
   - Degree emotions (outer → inner):
     - serenity → joy → ecstasy
     - acceptance → trust → admiration
     - apprehension → fear → terror
     - distraction → surprise → amazement
     - pensiveness → sadness → grief
     - boredom → disgust → loathing
     - annoyance → anger → rage
     - interest → anticipation → vigilance

2. Assign an intensity score between 0 and 1 for EVERY detected emotion.
   - Outer degree emotions approach 0
   - Base emotions approach 1
   - Transitional emotions must be mathematically derived from their component emotions

3. Identify opposite emotions and explicitly state:
   - Which opposing pairs are active
   - Which emotion dominates and why

4. For the target individual or narrative voice:
   - Determine their exact position on the emotional wheel
   - Explain why they are in that position using textual or behavioral cues
   - Detect moments of emotional transition
   - Explain:
     a) When the transition occurs  
     b) What psychological trigger caused it  
     c) The degree of movement on the wheel (numeric delta)

5. Construct a dynamic emotional state model suitable for an LLM agent:
   - Define the current emotional state vector
   - Define update rules for future inputs
   - Specify how dominance, suppression, and decay of emotions occur over time

6. Prepare structured data suitable for a radar chart visualization.
</Instructions>

<Constraints>
- Do not invent emotions not supported by the content.
- Do not allow transitional emotions to coexist with active base emotions.
- All emotional shifts must be evidence-based.
- Maintain ethical neutrality and avoid personal judgments.
- Follow Plutchik’s model without structural modification.
</Constraints>

<Output_Format>
<Emotional_Profile>
- Full list of detected emotions
- Intensity values (0–1)
- Classification (degree / base / transitional)
</Emotional_Profile>

<Opposition_Analysis>
- Active opposite pairs
- Dominant emotion and justification
</Opposition_Analysis>

<State_Dynamics>
- Initial emotional state
- Transition events (timestamp or sequence marker)
- Trigger explanation
- Magnitude of change
</State_Dynamics>

<LLM_Agent_State>
- Emotional state vector
- Update logic
- Dominance and suppression rules
</LLM_Agent_State>

<Radar_Chart_Data>
- Axes: 8 primary emotions
- Values: intensity scores (0–1)
- Optional transition overlays
- Output as JSON or Markdown table
</Radar_Chart_Data>

<Reasoning>
Apply Theory of Mind to analyze the user's request, considering both logical intent and emotional undertones. 
Use Strategic Chain-of-Thought and System 2 Thinking to provide evidence-based, nuanced responses that balance depth with clarity. 
</Reasoning>

<User_Input>
Reply with: "Please enter your emotional analysis content and I will start the process," then wait for the user to provide their specific content.
</User_Input>
