
from pydantic import BaseModel, Field
from typing import Optional
 
 
class PersonalitySection(BaseModel):
    """Base class for all personality sections."""
    section_name: str
    questions: list[str]
    answers: dict[str, Optional[str]] = Field(default_factory=dict)
 
    def model_post_init(self, __context):
        # Pre-populate answers dict with None for each question
        self.answers = {q: None for q in self.questions}
 
    def unanswered(self) -> list[str]:
        return [q for q, a in self.answers.items() if a is None]
 
    def answer(self, question: str, response: str):
        if question in self.answers:
            self.answers[question] = response
        else:
            raise ValueError(f"Question not found in section '{self.section_name}': {question}")
 
    def is_complete(self) -> bool:
        return all(v is not None for v in self.answers.values())
 
 
class IdentityAndSelfConcept(PersonalitySection):
    section_name: str = "Identity and Self-Concept"
    questions: list[str] = [
        "How would you describe your personality in three words?",
        "What values are most important in your life?",
        "What motivates your decisions?",
        "How do you define success?",
        "What role do you typically play in a group?",
    ]
 
 
class CognitiveStyleAndCuriosity(PersonalitySection):
    section_name: str = "Cognitive Style and Curiosity (Openness)"
    questions: list[str] = [
        "Do you enjoy exploring new ideas and concepts?",
        "How often do you try new experiences?",
        "What topics fascinate you the most?",
        "Do you prefer routine or novelty?",
        "How important is creativity in your life?",
    ]
 
 
class DisciplineAndOrganization(PersonalitySection):
    section_name: str = "Discipline and Organization (Conscientiousness)"
    questions: list[str] = [
        "How do you plan your goals?",
        "Are you more spontaneous or structured?",
        "How do you handle deadlines?",
        "What habits define your daily routine?",
        "How do you stay organized?",
    ]
 
 
class SocialOrientation(PersonalitySection):
    section_name: str = "Social Orientation (Extraversion)"
    questions: list[str] = [
        "Do you gain energy from social interaction or solitude?",
        "How often do you initiate conversations with others?",
        "Do you enjoy large gatherings or small groups?",
        "How comfortable are you speaking publicly?",
        "What role do friendships play in your life?",
    ]
 
 
class CooperationAndEmpathy(PersonalitySection):
    section_name: str = "Cooperation and Empathy (Agreeableness)"
    questions: list[str] = [
        "How do you respond to disagreements?",
        "How important is helping others to you?",
        "How do you resolve conflicts?",
        "How do you react when someone criticizes you?",
        "How do you build trust with people?",
    ]
 
 
class EmotionalStability(PersonalitySection):
    section_name: str = "Emotional Stability (Neuroticism)"
    questions: list[str] = [
        "How do you respond to stressful situations?",
        "How often do you worry about the future?",
        "How do you manage negative emotions?",
        "What situations make you anxious?",
        "How do you recover from setbacks?",
    ]
 
 
class InterestsAndLifestyle(PersonalitySection):
    section_name: str = "Interests and Lifestyle"
    questions: list[str] = [
        "What hobbies occupy most of your time?",
        "What media do you consume most often?",
        "What topics do you enjoy discussing online?",
        "What causes or issues matter to you?",
        "What type of environment helps you thrive?",
    ]
 
 
class DecisionMakingAndValues(PersonalitySection):
    section_name: str = "Decision-Making and Values"
    questions: list[str] = [
        "Do you rely more on logic or intuition?",
        "What principles guide your decisions?",
        "How do you evaluate risks?",
        "Do you prefer independence or collaboration?",
        "What long-term goals shape your choices?",
    ]
 
 
class SocialMediaBehavior(PersonalitySection):
    section_name: str = "Social Media Behavior"
    note: str = "These are specifically useful for AI personality inference."
    questions: list[str] = [
        "What kinds of posts do you share most frequently?",
        "How often do you engage in discussions online?",
        "What topics trigger your strongest reactions?",
        "How do you respond to disagreement on social media?",
        "What motivates you to share something online?",
    ]
 
 
class ReflectionAndIdentityEvolution(PersonalitySection):
    section_name: str = "Reflection and Identity Evolution"
    questions: list[str] = [
        "How has your personality changed over time?",
        "What experiences shaped who you are today?",
        "What lessons have most influenced your outlook?",
        "What do people misunderstand about you?",
        "What legacy do you want to leave behind?",
    ]
 
 
# ── Registry ──────────────────────────────────────────────────────────────────
 
ALL_SECTIONS: list[PersonalitySection] = [
    IdentityAndSelfConcept(),
    CognitiveStyleAndCuriosity(),
    DisciplineAndOrganization(),
    SocialOrientation(),
    CooperationAndEmpathy(),
    EmotionalStability(),
    InterestsAndLifestyle(),
    DecisionMakingAndValues(),
    SocialMediaBehavior(),
    ReflectionAndIdentityEvolution(),
]
 
 
# ── Example iteration usage ───────────────────────────────────────────────────
 
def ask_llm(question: str, section_name: str) -> str:
    """
    Replace this stub with your actual LLM call, e.g.:
        client.messages.create(...)
    The function receives the question and its section for context.
    """
    raise NotImplementedError("Wire up your LLM client here.")
 
 
def run_interview(sections: list[PersonalitySection] = ALL_SECTIONS) -> list[PersonalitySection]:
    """
    Iterates through every section and every question, calls the LLM,
    and stores the response back on the model.
    """
    for section in sections:
        print(f"\n{'='*60}")
        print(f"Section: {section.section_name}")
        print('='*60)
 
        for question in section.questions:
            print(f"\nQ: {question}")
            response = ask_llm(question, section.section_name)
            section.answer(question, response)
            print(f"A: {response}")
 
    return sections
 
 
if __name__ == "__main__":
    run_interview()