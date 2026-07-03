# ADAPTERS 

## Llama 4 dedicated endpoint 0.398 per minute with 15 minute auto shutdown
<!-- from together import Together

client = Together()

response = client.endpoints.create(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP4",
    display_name="e.woods.business@icloud.com: meta-llama - Llama-4-Maverick-17B-128E-Instruct-FP4",
    hardware="2x_nvidia_b200_180gb_sxm",
    min_replicas=1,
    max_replicas=1
)
print(response) -->



### Data completion format: SFT 
{"prompt": "...", "completion": "..."}
{"prompt": "...", "completion": "..."}


### Conversational message format with weighting (long messages are chunked into prompt completion with weights when a user supplies feedback for response preferences): 
### Zero is completely masked, 1 is preferred
{
  "messages": [
    {"role": "user", "content": "Can you explain machine learning?", "weight": 0},
    {"role": "assistant", "content": "Machine learning is...", "weight": 1}
  ],
  "weight": 0.9
}
{
  "messages": [
    {"role": "user", "content": "Can you explain why?", "weight": 0},
    {"role": "assistant", "content": "I can't", "weight": 1}
  ],
  "weight": 0.1
}


---


# LoRA training: 
<!-- https://docs.together.ai/docs/fine-tuning/lora-vs-full -->

# Deploying a model: 
<!-- https://docs.together.ai/docs/fine-tuning/deployment -->


# Pricing Estimation: 
## https://docs.together.ai/docs/fine-tuning/pricing

<!-- import os

from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

estimate = client.fine_tuning.estimate_price(
    training_file="file-abc123",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    n_epochs=3,
    training_method={"method": "sft"},
    training_type={"type": "Lora", "lora_r": 8},
)

print(estimate) -->

# Python sdk: https://github.com/togethercomputer/together-py