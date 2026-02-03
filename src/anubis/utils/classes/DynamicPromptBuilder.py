
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.anubis.utils.prompts.system_prompts import IDENTITY_SYSTEM_PROMPT_TEMPLATE

class DynamicPromptBuilder:
    """Builder for creating prompts with optional components."""
    
    def __init__(self, base_prompt: str = IDENTITY_SYSTEM_PROMPT_TEMPLATE):
        self.base_prompt = base_prompt

    # Base identity rendering function
    def render_identity_context(self, ai_context: Dict[str, Any]) -> str:
        """
        Recursively render AI context dictionary into natural language.

        Args:
            ai_context: Dictionary containing identity metadata

        Returns:
            Formatted string representing the AI's identity
        """
        if not ai_context:
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

        return format_dict(ai_context)
    
    def build_prompt(
        self,
        ai_context: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        retrieved_docs: Optional[str] = None,
        system_time: Optional[str] = None,
        temporary_message: Optional[str] = None
    ) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate with optional components.
        
        Args:
            ai_context: Dictionary of AI identity metadata
            user_context: Information about current user
            retrieved_docs: Retrieved document context
            system_time: Current timestamp
            temporary_message: Optional temporary instruction/context
            
        Returns:
            ChatPromptTemplate ready for invocation
        """
        # Render AI context
        ai_context_str = self.render_identity_context(ai_context or {})
        
        # Build user context
        user_context_str = user_context or "User identity unknown."
        
        # Build retrieved docs
        retrieved_docs_str = retrieved_docs or "No additional documents retrieved."
        
        # Build system time
        system_time_str = system_time or "Time information unavailable."
        
        # Build temporary message (this is the key part - it's optional!)
        if temporary_message:
            temp_msg_str = f"\n=== IMMEDIATE CONTEXT ===\n{temporary_message}\n"
        else:
            temp_msg_str = ""  # Empty string if no temporary message
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.base_prompt),
            ("placeholder", "{messages}"),
        ])
        

        # Mapping of the variables to the values for injection into the system prompt template
        prompt_vars = {
            "ai_context": ai_context_str,
            "user_context": user_context_str,
            "retrieved_docs": retrieved_docs_str,
            "system_time": system_time_str,
            "temporary_message": temp_msg_str
        }

        return prompt, prompt_vars

