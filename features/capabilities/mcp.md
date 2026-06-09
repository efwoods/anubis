# create an MCP server from your linux os to have langgraph query your filesystem

To enable LangGraph to query your filesystem, you can use the official @modelcontextprotocol/server-filesystem MCP server. This server acts as a standardized "USB-C" port for AI agents, allowing LangGraph to read, write, and search your local Linux files within secured, restricted directories.Step 1: Install the MCP Filesystem ServerThe official filesystem server is built on Node.js and distributed via npm. You can install and run it directly using npx.Run the following command in your Linux terminal to ensure it works, substituting /your/allowed/path with the specific directory (e.g., /home/username/documents) you want LangGraph to interact with:bashnpx -y @modelcontextprotocol/server-filesystem /your/allowed/path
Use code with caution.Note: This command starts the server in stdio (Standard I/O) mode, which is the default way LangGraph connects to MCP tools.Step 2: Integrate the Server into LangGraphIn your Python environment, use the langchain-mcp-adapters package to connect your LangGraph agent to the running MCP server.Here is a minimal script that initializes the MCP client, fetches the filesystem tools, and passes them to a LangGraph ReAct agent:pythonimport asyncio
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

## 1. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

## 2. Configure the MCP Server using STDIO
mcp_config = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/your/allowed/path"]
        }
    }
}

async def run_agent():
    # 3. Connect to the MCP server
    async with MultiServerMCPClient(mcp_config) as client:
        # Fetch the filesystem tools
        tools = await client.get_tools()
        
        # 4. Create your LangGraph agent with the MCP tools
        agent = create_react_agent(llm, tools=tools)
        
        # 5. Query the agent
        state = {"messages": [("user", "List all files in the directory and find the one with 'config' in its name")]}
        result = await agent.ainvoke(state)
        
        print(result["messages"][-1].content)

## Run the async loop
asyncio.run(run_agent())
Use code with caution.Step 3: Ensure Dependencies are InstalledMake sure you have the necessary libraries and your OpenAI API key set:bash# Install LangGraph and MCP adapters
pip install langgraph langchain-openai langchain-mcp-adapters

## Set your API key
export OPENAI_API_KEY="your-api-key-here"
Use code with caution.If you want to tailor this implementation, please tell me:What specific files or types of queries do you want LangGraph to run?Are you using Docker, or do you prefer a local Node.js/Python environment?I can provide specific adjustments to the directory permissions or node workflows.YouTube·LangChainUsing MCP with LangGraph agentsFeb 19, 2025 — this is lson lang chain connecting LMS to different sources of context. like tools like data sources is notoriously challenging an...6:00Neo4jQuickly build a React agent with LangGraph and MCP - Neo4jAug 19, 2025 — Imports and setup We first import the required libraries. We'll primarily use opens in new tabLangChain and opens in new tabLangGr...GitHubPart 5 — MCP Integration with LangGraph - Shafiqul AIDec 3, 2024 — In this project, MultiServerMCPClient from langchain-mcp-adapters handles the connection. Instantiate it directly, then call await...14 sites


 