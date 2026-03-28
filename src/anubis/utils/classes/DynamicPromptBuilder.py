
from typing import Optional, Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from src.anubis.utils.prompts.system_prompts import IDENTITY_SYSTEM_PROMPT_TEMPLATE

from langchain_core.documents import Document

class DynamicPromptBuilder:
    """Builder for creating prompts with optional components."""
    
    def __init__(self, base_prompt: str = IDENTITY_SYSTEM_PROMPT_TEMPLATE):
        self.base_prompt = base_prompt

    # Base identity rendering function
    def render_identity_context(self, assistant_context: Dict[str, Any]) -> str:
        """
        Recursively render AI context dictionary into natural language.

        Args:
            assistant_context: Dictionary containing identity metadata

        Returns:
            Formatted string representing the AI's identity
        """
        if not assistant_context:
            return "No specific identity information available."

        def format_dict(d: Dict[str, Any], indent: int = 0) -> str:
            """Recursively format nested dictionaries."""
            lines = []
            prefix = "  " * indent

            for key, value in d.items():
                # Convert key from snake_case to Title Case
                readable_key = key.replace('_', ' ').title()

                if isinstance(value, dict):
                    lines.append(f"{prefix}{readable_key}:")
                    lines.append(format_dict(value, indent + 1))
                elif isinstance(value, list):
                    lines.append(f"{prefix}{readable_key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(format_dict(item, indent + 1))
                        else:
                            lines.append(f"{prefix}  - {item}")
                elif value is not None and value != "":
                    lines.append(f"{prefix}{readable_key}: {value}")

            return "\n".join(lines)

        return format_dict(assistant_context)
    
    def build_prompt(
        self,
        assistant_name: Optional[str] = None,
        assistant_description: Optional[str] = None,
        assistant_identity: Optional[List[Document]] = None,
        assistant_emotions: Optional[List[Document]] = None,
        # assistant_context: Optional[Dict[str, Any]] = None,
        # user_context: Optional[Dict[str, Any]] = None,
        retrieved_knowledge: Optional[List[Document]] = None,
        retrieved_memories: Optional[List[Document]] = None,
        direct_quotes: Optional[List[Document]] = None,
        user_name: Optional[str] = None,
        user_description: Optional[str] = None,
        user_identity: Optional[List[Document]] = None,
        user_emotions: Optional[List[Document]] = None,
        system_time: Optional[str] = None,
    ) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate with optional components.
        
        Args:
            assistant_context: Dictionary of AI identity metadata
            user_context: Information about current user
            retrieved_knowledge: Retrieved document context
            system_time: Current timestamp
            temporary_message: Optional temporary instruction/context
            
        Returns:
            ChatPromptTemplate ready for invocation
        """

        # Render Assistant Name:
        if assistant_name is None or assistant_name is "":
            # assistant_name = "You don't know your name."    
            assistant_name = ""

        # Render Assistant Identity
        if assistant_identity is None or len(assistant_identity) == 0:
            if assistant_description is None:
                # assistant_identity_str = "You don't have any information about identity."
                assistant_identity_str = ""
            else:
                assistant_identity_str = assistant_description
        else:
            assistant_identity_str = "\n\n".join([doc.page_content for doc in assistant_identity])
            if assistant_description is not None:
                assistant_identity_str = assistant_description + "\n\n" + assistant_identity_str
        
        if direct_quotes is not None and len(direct_quotes) != 0:
            direct_quotes_str = "\n\n".join([doc.page_content for doc in direct_quotes])
        else:
            direct_quotes_str = ""

        # Render AI context
        # assistant_context_str = self.render_identity_context(assistant_context or {})

        if user_name is None or user_name == '':
            user_name = ""
        

        if user_identity is None or len(user_identity) == 0:
            if user_description is None:
                # user_identity_str = "You don't have any information about identity of the person or people you are communicating with."
                user_identity_str = ""
            else:
                user_identity_str = user_description
        else:
            user_identity_str = "\n\n".join([doc.page_content for doc in user_identity])
            if user_description is not None:
                user_identity_str = user_description + "\n\n" + user_identity_str
        
        # Build user context
        # user_context_str = user_context or "User identity unknown."
        
        # Build retrieved knowledge
        if retrieved_knowledge is not None and len(retrieved_knowledge) != 0:
            retrieved_knowledge_str = "\n\n".join([doc.page_content for doc in retrieved_knowledge])
        else:
            # retrieved_knowledge_str = "No additional documents retrieved."
            retrieved_knowledge_str = ""
        
        # Build retrieved memories (associated memories given the conversation)
        if retrieved_memories is None:
            # retrieved_memories_str = "No additional memories retrieved."
            retrieved_memories_str = ""
        else:
            retrieved_memories_str = "\n\n".join([doc.page_content for doc in retrieved_memories])

        if assistant_emotions is None:
            # assistant_emotions_str = "Unaware of current emotions of self."
            assistant_emotions_str = ""

        if user_emotions is None:
            # user_emotions_str = "Unaware of the current emotions of the person or people you are addressing."
            user_emotions_str = ""

        # Build system time
        if system_time is None:
            system_time = "Time information unavailable."
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.base_prompt),
        ])
        

        # Mapping of the variables to the values for injection into the system prompt template
        prompt_vars = {
            "assistant_name": assistant_name, 
            "assistant_identity": assistant_identity_str,
            "assistant_emotions": assistant_emotions_str,
            "retrieved_knowledge": retrieved_knowledge_str,
            "retrieved_memories": retrieved_memories_str,
            "direct_quotes": direct_quotes_str,
            "user_name": user_name, 
            "user_identity": user_identity_str,
            "user_emotions": user_emotions_str,
            "system_time": system_time,
        }

        populated_template =  prompt.invoke(prompt_vars)

        return populated_template

