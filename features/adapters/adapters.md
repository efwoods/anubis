train and adapter adapters for cost effective use (when there is substantial finances and quality of available data (collect data in preparation; measure average improvement given an adapter and use cases))

# example application with meta-llama model adapters
https://github.com/efwoods/adapter-testing
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8
https://llama.developer.meta.com/docs/overview/?team_id=3775658676066270&project_id=1839029780145302

# state of the art
https://developers.openai.com/api/docs/guides/model-optimization

# potential fine-tuning source (unlikely):
https://www.together.ai/fine-tuning

# Train adapters using vLLM, reward functions, direct quotes, and GRPO implementing those reward functions to achieve training that is pertinent to the measured metrics responsible for judging authenticity; Training with respect to these metrics as reward functions will allow for higher quality responses and may not require larger language models for increased performance (reduces cost; required vLLM training server, upload adapter to inference provider after training if not self-serving inference). Otherwise using DPO and SFT to train using LLMs that require either a large user base to break even monthly (2000 users and $9 for 24/7 inference). 
# Risk: The cost of cloud compute must be less than the cost to train with the inference provider and will take time to develop. Quality of a smaller parameter model may not meet the quality of a multi-billion parameter model. 
# Advantage: using the same techniques on smaller models may be applicable to large language models and creating the process an infastructure will save time and money in the future; 

# Risk: the benefits of training GRPO, SFT, DPO on a multi-billion parameter LLM with respect to the advantage of quality is uncertain and costly with respect to time and financial-resources. 


# Fireworks.ai GRPO with custom reward function: 
https://docs.fireworks.ai/fine-tuning/training-api/cookbook/rl
https://github.com/fw-ai/cookbook/tree/main/training
https://docs.fireworks.ai/fine-tuning/deploying-loras

<!-- from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import RolloutSample

def make_rollout_fn(setup):
    sampler = build_deployment_sampler(setup)
    sample_kwargs = dict(setup.sample_kwargs)

    async def rollout_fn(sample_prompt: dict) -> RolloutSample | None:
        completions = await sampler.sample_with_prompt_tokens(
            sample_prompt["prompt_token_ids"], n=1, **sample_kwargs,
        )
        if not completions:
            return None
        c = completions[0]
        output = list(c.full_tokens)[c.prompt_len:]
        return RolloutSample(
            tokens=list(c.full_tokens),
            logprobs=[0.0] * c.prompt_len + list(c.inference_logprobs),
            loss_mask=[0] * c.prompt_len + [1] * len(output),
            reward=score(c),                       # your reward function
            finish_reason=c.finish_reason,
            text=c.text,
        )

    return rollout_fn -->

https://huggingface.co/docs/trl/en/grpo_trainer 
