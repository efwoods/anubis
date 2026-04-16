from typing import List
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.anubis.utils.model import init_model



""" LLAMA 4 ADAPTER TRAINING FORMAT """

async def llm_single_turn_dataset(question_list: List[str], answer_list: List[str]) -> List[dict]:
    """ Creates a Messages Dataset of Single Turns for a list of question and answer pairs. Used for LLM Adapter Training Format."""
    single_turn_dataset = []
    for question, answer in zip(question_list, answer_list):
        turn = {"messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]}
        single_turn_dataset.append(turn)
    return single_turn_dataset

def llm_multiturn_dataset_one_conversation(question_list: List[str], answer_list: List[str]) -> dict:
    """ Creates a Messages Dataset of a conversation of question and answer pairs. Used for LLM Adapter Training Format.
        This is a single conversation. A list of multiple conversations must be used to for the entire final dataset.
    """
    list_of_messages = []
    for question, answer in zip(question_list, answer_list):
        turn = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        list_of_messages += turn
    multi_turn_dataset = {"messages": list_of_messages}
    return multi_turn_dataset

""" LANGSMITH DATASET FORMAT """

async def langsmith_dataset(question_list: List[str], answer_list: List[str], dataset_source_filename: str) -> List[dict]:
    """ Creates a list of dict example question and answer inputs and outputs """
    examples = []
    examples.append({
        "inputs":{"question": question}, 
        "outputs": {"answer":answer}, 
        "metadata": {"source", dataset_source_filename}} for question, answer in zip(question_list, answer_list))
    return examples

async def create_question_list(str_messages_list: list[str]) -> List[str]:
    class GeneratedQuestionsList(BaseModel):
        question_list: List[str]

    human_messages_list = [HumanMessage(content=message_str) for message_str in str_messages_list]

    # TODO: CALCULATE TOKEN USAGE response['response_metadata']

    model_with_structured_output = init_model(model_without_tools=False, response_format= GeneratedQuestionsList)

    system_message = SystemMessage(content="Given this list of messages, generate a query to which the message is the response. THERE MUST BE A QUESTION FOR RESPONSE AND THE QUESTION ORDER IN THE LIST MUST MATCH THE RESPONSE ORDER. These questions must be succinct.")
    
    messages = [system_message] + human_messages_list

    response = await model_with_structured_output.ainvoke(input=messages)

    return response.question_list
