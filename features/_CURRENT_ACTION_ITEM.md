# collect token usage metrics

# response evaluation loop with langsmith dataset from langsmith import RunEvalConfig(evaluators=["sentiment", "cot_qa","bleu", "rouge"], custom_evaluators=[RunEvalConfig.Criteria({"authenticity":"AUTHENTICITY_PROMPT (were these text written by the same author?)"}), eval_llm="gpt-5.4-nano"])

<!-- https://reference.langchain.com/python/langchain-classic/smith/evaluation/config/RunEvalConfig -->
<!-- https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/qa-correctness/qa-correctness.ipynb -->

# Clean semantic chunking (use defined context window with overlap in process documents to text)
# clean conversation summaries
