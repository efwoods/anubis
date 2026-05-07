# 05/06/2026
# CLASSIFYING CONETNT (TEXT IS CLASSIFIED AS DIALOGUE)
# PRE-PROCESSING CSV (LIST OF TWEETS NEED TO BE PREPROCESSED INTO JSON FORMAT FOR CLASSIFICATION ON A PORTION OF THE JSON STRING)


# TOTAL PRIORITIES
# CURRENT PRIORITIES:
# health data analysis

# Test image urls
# process upload images that are not reference images
# process upload text documents that are autobiographical (needs facts extracted and re-written using LLM; facts must not change but the wording changes to prevent lawsuit: https://lexfridman.com/, https://www.hubermanlab.com/about, https://en.wikipedia.org/wiki/Lex_Fridman; extract facts and quote from wikis: https://fallout.fandom.com/wiki/Curie)
# process upload text documents that are quotes/tweets: https://x.com/lexfridman
# process upload text documents that are dialogues: https://www.youtube.com/watch?v=IvqRSEP_o-o
# process upload text documents that are monologues: (monologue of target only: https://www.youtube.com/watch?v=gIF_D6iUusU&list=PL9rU625vkl4XSOQDZxdFhVZgows3FJiKH&index=4) (monologue with target followed by non-target monologue: https://www.youtube.com/watch?v=-tQwzhHjAVI&list=PL9rU625vkl4XSOQDZxdFhVZgows3FJiKH&index=3)
# process upload text documents that are multi-speaker with the target
# process upload text documents that are multi-speaker that reference the target
# process upload text documents that are multi-speaker that reference the target with the target direct quotes: (https://www.youtube.com/watch?v=Px_5Z0pPlPc&t=2s) autobiographical about the target with direct quotes and multi-speaker


# Integrate reference images from urls in the upload media endpoint


# Integrate reference audio from personal media and urls in the upload media endpoint

#####


# document refactor

# collect token usage metrics

# response evaluation loop with langsmith dataset from langsmith import RunEvalConfig(evaluators=["sentiment", "cot_qa","bleu", "rouge"], custom_evaluators=[RunEvalConfig.Criteria({"authenticity":"AUTHENTICITY_PROMPT (were these text written by the same author?)"}), eval_llm="gpt-5.4-nano"])

<!-- https://reference.langchain.com/python/langchain-classic/smith/evaluation/config/RunEvalConfig -->
<!-- https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/qa-correctness/qa-correctness.ipynb -->

# Clean semantic chunking (use defined context window with overlap in process documents to text)
# clean conversation summaries

# VERY IMPORTANT:
# Reduce latency to less than 2 seconds
# Latency Reduction


# process a link tree: https://linktr.ee/ev0ra?utm_source=linktree_profile_share&ltsid=217800c7-7f92-45dd-88f6-9f364d64275c
# deep research agent: (https://docs.langchain.com/oss/python/deepagents/deep-research#build-a-deep-research-agent); https://github.com/langchain-ai/open_deep_research; https://academy.langchain.com/courses/take/deep-research-with-langgraph/texts/67644896-getting-set-up; https://github.com/langchain-ai/deep_research_from_scratch