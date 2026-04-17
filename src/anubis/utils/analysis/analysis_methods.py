from src.anubis.utils.prompts.psycho_analysis.OCEAN_analysis import (
    OPENNESS_OCEAN_ANALYSIS_PROMPT, 
    CONSCIENTIOUSNESS_OCEAN_ANALYSIS_PROMPT, 
    EXTRAVERSION_OCEAN_ANALYSIS_PROMPT, 
    AGREEABLENESS_OCEAN_ANALYSIS_PROMPT, 
    NEUROTICISM_OCEAN_ANALYSIS_PROMPT)

from src.anubis.utils.prompts.psycho_analysis.OCEAN_analysis import (
    OPENNESS_OCEAN_ANALYSIS_EXTRACTION, 
    CONSCIENTIOUSNESS_OCEAN_ANALYSIS_EXTRACTION, 
    EXTRAVERSION_OCEAN_ANALYSIS_EXTRACTION, 
    AGREEABLENESS_OCEAN_ANALYSIS_EXTRACTION, 
    NEUROTICISM_OCEAN_ANALYSIS_EXTRACTION)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.anubis.utils.model import init_model
from langchain_core.documents import Document
from uuid import uuid4

import logging
logger = logging.getLogger(__name__)

""" OCEAN ANALYSIS """
from typing import Optional
async def perform_ocean_analysis(human_message: HumanMessage, additional_metadata = Optional[dict]):
    """  This function ingests a human message containing 
    content to analyze and returns a list of documents 
    containing the five results of the analysis."""

    logger.info("breakpoint")
    # TODO: CALCULATE TOKEN USAGE response['response_metadata']

    openness_model_ocean_analysis = init_model(model_without_tools=False, response_format=OPENNESS_OCEAN_ANALYSIS_EXTRACTION)
    conscientiousness_model_ocean_analysis = init_model(model_without_tools=False, response_format=CONSCIENTIOUSNESS_OCEAN_ANALYSIS_EXTRACTION)
    extraversion_model_ocean_analysis = init_model(model_without_tools=False, response_format=EXTRAVERSION_OCEAN_ANALYSIS_EXTRACTION)
    agreeableness_model_ocean_analysis = init_model(model_without_tools=False, response_format=AGREEABLENESS_OCEAN_ANALYSIS_EXTRACTION)
    neuroticism_model_ocean_analysis = init_model(model_without_tools=False, response_format=NEUROTICISM_OCEAN_ANALYSIS_EXTRACTION)

    openness_ocean_analysis_prompt = SystemMessage(content=OPENNESS_OCEAN_ANALYSIS_PROMPT)
    conscientiousness_ocean_analysis_prompt = SystemMessage(content=CONSCIENTIOUSNESS_OCEAN_ANALYSIS_PROMPT)
    extraversion_ocean_analysis_prompt = SystemMessage(content=EXTRAVERSION_OCEAN_ANALYSIS_PROMPT)
    agreeableness_ocean_analysis_prompt = SystemMessage(content=AGREEABLENESS_OCEAN_ANALYSIS_PROMPT)
    neuroticism_ocean_analysis_prompt = SystemMessage(content=NEUROTICISM_OCEAN_ANALYSIS_PROMPT)

    openness_ocean_analysis_results = await openness_model_ocean_analysis.ainvoke([openness_ocean_analysis_prompt, human_message])
    conscientiousness_ocean_analysis_results = await conscientiousness_model_ocean_analysis.ainvoke([conscientiousness_ocean_analysis_prompt, human_message])
    extraversion_ocean_analysis_results = await extraversion_model_ocean_analysis.ainvoke([extraversion_ocean_analysis_prompt, human_message])
    agreeableness_ocean_analysis_results = await agreeableness_model_ocean_analysis.ainvoke([agreeableness_ocean_analysis_prompt, human_message])
    neuroticism_ocean_analysis_results = await neuroticism_model_ocean_analysis.ainvoke([neuroticism_ocean_analysis_prompt, human_message])

    results = [
    Document(page_content=openness_ocean_analysis_results.openness_description, metadata={"score": openness_ocean_analysis_results.openness, "reasons":openness_ocean_analysis_results.openness_reasons_and_examples, "processing_task_id": str(uuid4()), "document_id": str(uuid4()) }, id=str(uuid4())),
    Document(page_content=conscientiousness_ocean_analysis_results.conscientiousness_description, metadata={"score": conscientiousness_ocean_analysis_results.conscientiousness, "reasons":conscientiousness_ocean_analysis_results.conscientiousness_reasons_and_examples, "processing_task_id": str(uuid4()), "document_id": str(uuid4()) }, id=str(uuid4())),
    Document(page_content=extraversion_ocean_analysis_results.extraversion_description, metadata={"score": extraversion_ocean_analysis_results.extraversion, "reasons":extraversion_ocean_analysis_results.extraversion_reasons_and_examples, "processing_task_id": str(uuid4()), "document_id": str(uuid4()) }, id=str(uuid4())),
    Document(page_content=agreeableness_ocean_analysis_results.agreeableness_description, metadata={"score": agreeableness_ocean_analysis_results.agreeableness, "reasons":agreeableness_ocean_analysis_results.agreeableness_reasons_and_examples, "processing_task_id": str(uuid4()), "document_id": str(uuid4()) }, id=str(uuid4())),
    Document(page_content=neuroticism_ocean_analysis_results.neuroticism_description, metadata={"score": neuroticism_ocean_analysis_results.neuroticism, "reasons":neuroticism_ocean_analysis_results.neuroticism_reasons_and_examples, "processing_task_id": str(uuid4()), "document_id": str(uuid4()) }, id=str(uuid4()))]

    if additional_metadata:
        [doc.metadata.update(metadata=additional_metadata) for doc in results]

    return results




