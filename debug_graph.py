import asyncio
from langchain_core.messages import HumanMessage
from src.agent.graph import graph

async def main():
    config = {"configurable": {"thread_id": "vscode-debug"}}
    state = {
        "messages": [HumanMessage(content="Test message")],
        "is_last_step": False
    }

    # Hits breakpoints
    result = await graph.ainvoke(state, config)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())