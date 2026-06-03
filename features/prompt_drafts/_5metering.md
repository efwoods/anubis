# Token metrics:
I need to create three subscription tiers and metered usage of the api given the current subscription status beyond a limit (free-tier, pro-tier, premium). I need to aggregate all the token usage to understand how much I am spending and meter the customer usage. 
https://docs.stripe.com/api/billing/meter/object 

@metered_pipeline_for_request_use_and_subscription_tiers.md (1-3) 

Each call to any model needs to have the latency calculated as well as the input tokens, completion tokens, total tokens, model used, inference type (image model, model with structured output, inference model), total cost. These metrics need to be collected for each invocation from the response_metadata after the response has been converted to json (use time_ns before and after each call to call the model and a class should be defined to wrap each model creation and call to additionally return the metadata listed before. @/home/user/gh/anubis/ These model invocations are marked with # TODO: response_metrics_aggregation when the model is initialized and the actual call follows. 

This is an example of the response structure for every call of init_image_description_model (just focus on the invocation response format and returning a dict of the previous metrics along with accepting all original inputs and returning the original output as well whenever this type of model is used. 

@utility.py (41-129) 

This is an example of what needs to be collected for ANY model with response_format (model with structured output) even if not used for the "ContentSituationClassificationClass":
 @src/anubis/utils/classes/ContentSituationClassificationClass.py 


This is an example of a text call. Note that in the @src/anubis/utils/nodes.py during "respond" and "think" nodes @src/anubis/graph.py I will need to input the messages and examine the response if using a class wrapper. This is the example response format from AI message when calling with the following style:
{'content': 'In the hush of moonlight, a gentle unicorn tiptoed through a silver meadow, sprinkling stardust on sleepy flowers until the whole night felt safe and softly magical.',
 'additional_kwargs': {'refusal': None},
 'response_metadata': {'token_usage': {'completion_tokens': 39,
   'prompt_tokens': 132,
   'total_tokens': 171,
   'completion_tokens_details': {'accepted_prediction_tokens': 0,
    'audio_tokens': 0,
    'reasoning_tokens': 0,
    'rejected_prediction_tokens': 0},
   'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
  'model_provider': 'openai',
  'model_name': 'gpt-5.4-nano-2026-03-17',
  'system_fingerprint': None,
  'id': 'chatcmpl-DaXdtbSkEzZlaeQxmgngRi2A3r88D',
  'service_tier': 'default',
  'finish_reason': 'stop',
  'logprobs': None},
 'type': 'ai',
 'name': None,
 'id': 'lc_run--019de14e-568d-79f2-96d2-c41a890752e7-0',
 'tool_calls': [],
 'invalid_tool_calls': [],
 'usage_metadata': {'input_tokens': 132,
  'output_tokens': 39,
  'total_tokens': 171,
  'input_token_details': {'audio': 0, 'cache_read': 0},
  'output_token_details': {'audio': 0, 'reasoning': 0}}}

# Example text call:
@data_processing.ipynb (1-3) 

for each @src/api/webapp.py /message /message selected avatar, there should be the aggregration of totals for all the previous metrics for the single call (total cost, total input_tokens, total completion_tokens, total tokens, total latency_ms, and each of the previous categories for each distinct model type  and each distinct inference type)

# I need to implement rate limiting and respect rate limits of the models for which I am calling