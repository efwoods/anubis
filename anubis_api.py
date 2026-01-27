# send_message.py
from langgraph_sdk import get_client
import asyncio

client = get_client("http://localhost:2024")

async def send_message():
    input_msg