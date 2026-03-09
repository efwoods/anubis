
from src.anubis.utils.prompts.system_prompts import (
    EVALUATION_PROMPT_TEMPLATE, 
    RELEVANCY_SCORE_CRITERIA,
    RELEVANCY_SCORE_STEPS,
    COHERENCE_SCORE_CRITERIA,
    COHERENCE_SCORE_STEPS,
    CONSISTENCY_SCORE_CRITERIA,
    CONSISTENCY_SCORE_STEPS,
    FLUENCY_SCORE_CRITERIA,
    FLUENCY_SCORE_STEPS
)

from src.anubis.utils.model import init_model

from langchain_core.messages import SystemMessage

from rouge import Rouge

from bert_score import BERTScorer

async def get_llm_eval_scores(
        source_text: str,
        generated_response: str, 
        criteria: str = None,
        steps: str = None,
        metric_name: str = None, 
        normalization_scale_min: int = None,
        normalization_scale_max: int = None
): 
    if (criteria is None and 
        steps is None and 
        metric_name is None and 
        normalization_scale_max is None and 
        normalization_scale_min is None
    ):
        relevancy_system_prompt = SystemMessage(content = EVALUATION_PROMPT_TEMPLATE.format(
              criteria=RELEVANCY_SCORE_CRITERIA,
              steps=RELEVANCY_SCORE_STEPS,
              source_text=source_text,
              generated_response=generated_response,
              metric_name="Relevancy"
        ))

        coherence_system_prompt = SystemMessage(content = EVALUATION_PROMPT_TEMPLATE.format(
              criteria=COHERENCE_SCORE_CRITERIA,
              steps=COHERENCE_SCORE_STEPS,
              source_text=source_text,
              generated_response=generated_response,
              metric_name="Coherence"
        ))

        consistency_system_prompt = SystemMessage(content = EVALUATION_PROMPT_TEMPLATE.format(
              criteria=CONSISTENCY_SCORE_CRITERIA,
              steps=CONSISTENCY_SCORE_STEPS,
              source_text=source_text,
              generated_response=generated_response,
              metric_name="Consistency"
        ))

        fluency_system_prompt = SystemMessage(content = EVALUATION_PROMPT_TEMPLATE.format(
              criteria=FLUENCY_SCORE_CRITERIA,
              steps=FLUENCY_SCORE_STEPS,
              source_text=source_text,
              generated_response=generated_response,
              metric_name="Fluency"
        ))

        system_prompt_list = [
            relevancy_system_prompt, 
            coherence_system_prompt, 
            consistency_system_prompt, 
            fluency_system_prompt
        ]

        eval_name_list = ["Relevancy", "Coherence", "Consistency", "Fluency"]

        model = init_model()

        results = []
        for index in range(0, len(system_prompt_list)):
            response = await model.ainvoke(system_prompt_list[index])
            results.append(response.content)
        evaluation_results = {eval_name: int(result) for eval_name, result in zip(eval_name_list, results)}
        
        # Normalize 0 - 1
        evaluation_results['Relevancy'] = (evaluation_results['Relevancy'] - 1) / 4
        evaluation_results['Coherence'] = (evaluation_results['Coherence'] - 1) / 4
        evaluation_results['Consistency'] = (evaluation_results['Consistency'] - 1) / 4
        evaluation_results['Fluency'] = (evaluation_results['Fluency'] - 1) / 2

        return evaluation_results

    else:
        system_prompt = SystemMessage(content = EVALUATION_PROMPT_TEMPLATE.format(
              criteria=criteria,
              steps=steps,
              source_text=source_text,
              generated_response=generated_response,
              metric_name=metric_name,
              normalization_scale_min=normalization_scale_min, 
              normalization_scale_max = normalization_scale_max
        ))

        model = init_model()
        results = await model.ainvoke(system_prompt)

        # Normalized results from zero to one:
        norm_results = (int(results.content) - normalization_scale_min) / (normalization_scale_max - normalization_scale_min)

        return {metric_name: norm_results}


async def get_rouge_score(source_text: str, 
                           generated_response: str):

    rouge = Rouge()
    return rouge(source_text, generated_response)


async def get_bert_score(source_text: str, 
                          generated_response: str):
    
    scorer = BERTScorer(lang="en")

    precision, recall, f_score = scorer.score([source_text], [generated_response])
    return f_score
    
async def evaluate(source_text: str, generated_response: str):
    """
    Evaluate the llm generated response against the source text with respect to the following criteria:
    - RELEVANCY
    - COHERENCE
    - CONSISTENCY
    - FLUENCY
    - sentence structure using ROUGE score
    - semantic similarity using BERT score

    Args:
        source_text (str): ground truth of what the target ACTUALLY said.
        generated_response (str): generated response from mimicing LLM.
    """

    evaluation_results = {}
    semantic_similarity_bert_f_score = get_bert_score(source_text, generated_response)
    sentence_structure_rouge_f_score = get_rouge_score(source_text, generated_response)
    llm_eval_scores = get_llm_eval_scores(source_text, generated_response)

    eval_list = [
        {"semantic_similarity_bert_f_score": semantic_similarity_bert_f_score}, 
        {"sentence_structure_rouge_f_score": sentence_structure_rouge_f_score}, 
        llm_eval_scores
    ]

    for eval in eval_list:
        evaluation_results.update(eval)

    return evaluation_results