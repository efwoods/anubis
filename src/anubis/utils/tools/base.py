from typing import Dict, List, Optional
from langchain_core.tools import BaseTool

from src.anubis.utils.tools.identity.identity_tools import (
    learn_information_about_the_user, 
    update_self_identity_mem_from_user_txt, 
    recall_memories
)


# tools dictionary
all_tools = {
    "update_self_identity_mem_from_user_txt": update_self_identity_mem_from_user_txt,
    "learn_information_about_the_user":learn_information_about_the_user,       
    "recall_memories":recall_memories
}

identity_tools = {
    "update_self_identity_mem_from_user_txt": update_self_identity_mem_from_user_txt,
    "learn_information_about_the_user":learn_information_about_the_user,       
}

avatar_tools = {
    "recall_memories":recall_memories
}

def get_tools(tool_names: Optional[List[str]] = None) -> List[BaseTool]:


    if tool_names is None:
        return list(all_tools.values())
    elif tool_names[0] is "identity_tools":
        return list(identity_tools.values())
    elif tool_names[0] is "avatar_tools":
        return list(avatar_tools.values)
    
    return [all_tools[name] for name in tool_names if name in all_tools]

def get_tool_names(tools: Optional[str] = None):
    if tools is "identity":
        return [key for key, value in identity_tools]
    elif tools is "avatar":
        return [key for key, value in avatar_tools]
    return [key for key, value in all_tools]

def get_tools_by_name(tools: Optional[List[BaseTool]] = None) -> Dict[str, BaseTool]:
    if tools is None:
        tools = get_tools()

    return {tool.name: tool for tool in tools}
