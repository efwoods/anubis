from typing import Dict, List, Optional
from langchain_core.tools import BaseTool

def get_tools(tool_names: Optional[List[str]] = None) -> List[BaseTool]:

    # import tools
    from src.anubis.utils.tools.identity.identity_tools import 

    # tools dictionary
    all_tools = {
        
    }