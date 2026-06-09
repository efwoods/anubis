# **Architectural Frameworks for Behavioral Alignment, Validation, and State Recovery in Multi-Step Agentic Workflows**

## **Architectural Foundations of Stateful Recovery Loops**

Production-grade agentic systems require systematic architectures to mitigate behavioral anomalies, particularly when an autonomous agent fails to execute the correct action in a predefined workflow.1 When a transactional avatar, such as an automated order-placement system, makes an incorrect downstream call, restarts are computationally inefficient and degrade the user experience.1 Designing stateful recovery loops within a cyclic graph architecture provides a robust alternative.3 By defining an explicit, typed state schema that serves as the shared memory of the system, the workflow can track execution parameters, validation metrics, and historical feedback.3  
A highly effective paradigm is the Plan-Execute-Validate (PEV) pattern.2 This architecture isolates generation tasks from evaluation tasks by establishing dedicated processing and evaluation nodes.2 In this framework, a generator node proposes a state transition or a structured payload, while a subsequent validator node assesses the payload's compliance against strict operational schemas.2  
If the validator identifies an anomaly, a deterministic, non-LLM router node intercepts the workflow.2 This router bypasses probabilistic model calls, writes a detailed error explanation to the graph's state, increments a retry counter, and redirects execution back to the planner node.2

Python  
from typing import TypedDict, Annotated, Union, Dict, Any, List  
import operator  
from pydantic import BaseModel, Field, conlist  
from langgraph.graph import StateGraph, START, END  
from langgraph.checkpoint.memory import InMemorySaver

class OrderItem(BaseModel):  
    item\_name: str \= Field(..., min\_length=1, description="The name of the item being ordered.")  
    quantity: int \= Field(..., ge=1, le=100, description="The positive integer quantity.")

class StructuredOrder(BaseModel):  
    items: List\[OrderItem\] \= Field(..., description="List of ordered items.")  
    special\_instructions: Union\[str, None\] \= None

class AgentState(TypedDict):  
    raw\_input: str  
    parsed\_order: Union, None\]  
    is\_valid: bool  
    feedback: str  
    attempt\_count: int  
    validation\_history: Annotated\], operator.add\]

def order\_planner\_node(state: AgentState) \-\> Dict\[str, Any\]:  
    \# The planner ingests the raw input and any accumulated validation feedback   
    \# to construct or refine the structured order.  
    current\_attempt \= state.get("attempt\_count", 0\) \+ 1  
    return {"attempt\_count": current\_attempt}

def order\_validator\_node(state: AgentState) \-\> Dict\[str, Any\]:  
    \# The validator acts as the quality gate, confirming that the generated order   
    \# conforms to inventory rules and schema requirements.  
    order \= state.get("parsed\_order")  
    if not order or len(order.get("items",)) \== 0:  
        return {  
            "is\_valid": False,   
            "feedback": "Validation Failure: The order does not contain any valid items.",  
            "validation\_history": \[{"attempt": state\["attempt\_count"\], "error": "Empty items list"}\]  
        }  
    return {"is\_valid": True, "feedback": "Validation Passed."}

def execution\_router(state: AgentState) \-\> str:  
    \# A deterministic Python-based router decides whether to proceed to fulfillment or retry.  
    if state\["is\_valid"\]:  
        return "fulfill\_order"  
    if state\["attempt\_count"\] \>= 3:  
        return "escalate\_to\_human"  
    return "retry\_planning"

builder \= StateGraph(AgentState)  
builder.add\_node("order\_planner", order\_planner\_node)  
builder.add\_node("order\_validator", order\_validator\_node)

builder.set\_entry\_point("order\_planner")  
builder.add\_edge("order\_planner", "order\_validator")

builder.add\_conditional\_edges(  
    "order\_validator",  
    execution\_router,  
    {  
        "fulfill\_order": END,  
        "retry\_planning": "order\_planner",  
        "escalate\_to\_human": END  
    }  
)

checkpointer \= InMemorySaver()  
compiled\_graph \= builder.compile(checkpointer=checkpointer)

By ensuring that the state transitions are checkpointed after every node execution, this cyclic approach allows the agent to ingest failure modes directly into its context window during subsequent retries.2 The feedback loop forces the model to self-correct its planning parameters before committing actions to downstream APIs.2

## **Runtime Validation and Structured Error Mitigation**

Conversational and transactional systems are highly susceptible to Type I (False Positive) and Type II (False Negative) errors.6 These issues typically manifest as either the retrieval and output of incorrect information, or the failure to produce a designated semantic response in a scripted dialogue sequence.6

### **Mitigating Semantic False Negatives (Type II Errors)**

A semantic false negative occurs when the user provides an input that requires a specific response, but the agent fails to generate that exact response.6 For example, in an interactive scenario, a user might execute a gesture like *"I throw an empty gun"*, which should trigger the response *"You gave me an empty gun?"*.6 If the model generates a different phrase, it violates the expected dialogue tree.6  
To resolve these semantic errors, the validation layer can run a Natural Language Inference (NLI) model within the validating node.6 The NLI model treats the user's gesture as a premise and the model's generated output as a hypothesis.6 It then classifies the relationship into one of three logical categories:

* **Entailment**: The response logically follows the gesture.6  
* **Contradiction**: The response violates the context of the gesture.6  
* **Neutral**: The response is unrelated to the gesture.6

                     \+---------------------------+  
                     | User Input / Gesture:     |  
                     | "I throw an empty gun"    |  
                     \+-------------+-------------+  
                                   |  
                                   v  
                     \+---------------------------+  
                     |      LLM Generator        |  
                     \+-------------+-------------+  
                                   |  
                                   v  
                     \+---------------------------+  
                     | Generated Response:       |  
                     | "Here is some ammunition" |  
                     \+-------------+-------------+  
                                   |  
                                   v  
                     \+---------------------------+  
                     |    NLI Validation Gate    |  
                     \+-------------+-------------+  
                                   |  
                \+------------------+------------------+  
                |                                     |  
                v (Entailment)                        v (Contradiction / Neutral)  
      \+-------------------+                 \+----------------------------+  
      |  Pass to Output   |                 | Trigger Correction Policy: |  
      |       Gate        |                 | Overwrite with standard    |  
      \+-------------------+                 | output or run retry loop   |  
                                            \+----------------------------+

If the validation gate detects a contradiction or a neutral relationship, the system triggers a correction policy.7 Depending on how critical the conversation is, the system can either run a localized retry loop to generate a new response, or overwrite the incorrect phrase with the exact expected response.7

### **Mitigating Fact and Cardinality False Positives (Type I Errors)**

A false positive occurs when the model introduces incorrect or ungrounded information.6 For instance, if a user asks an agent for its favorite two colors and the model responds with only one color (such as *"green"*), it violates both factual accuracy and cardinality constraints.  
This failure mode is handled using strict schema enforcement with Pydantic.12 By configuring the model's output parser with a strict Pydantic schema, the model is forced to structure its response within a typed template.12 To enforce cardinality constraints, developers can use Pydantic field validators or constrained list structures:

Python  
from pydantic import BaseModel, Field, conlist

class ColorInquiryResponse(BaseModel):  
    favorite\_colors: conlist(str, min\_length=2, max\_length=2) \= Field(  
       ...,   
        description="Must contain exactly two favorite colors, representing red and blue."  
    )

If the model returns a single color or an incorrect value like *"green"*, the Pydantic parser raises a ValidationError.13 The validator node catches this exception, serializes the exact schema error details, and feeds them back into a localized retry loop.13  
When using this approach, developers often encounter a PydanticSerializationUnexpectedValue warning.14 This occurs during serialization if the system attempts to serialize the entire wrapper object returned by the framework instead of just the parsed model.14  
To resolve this, developers should call model\_dump() directly on the parsed Pydantic object and return only the raw dictionary.14

| Validation Framework | Processing Latency | Structural Capabilities | Integration Complexity | Primary Error Targets |
| :---- | :---- | :---- | :---- | :---- |
| **Pydantic Schema Validation** 12 | \< 2 ms 11 | Enforces exact data types, ranges, list sizes, and nested schemas 12 | Minimal; directly integrated via with\_structured\_output 12 | Cardinality errors, malformed JSON, and missing schema fields 7 |
| **Guardrails AI** 7 | \< 10 ms 11 | Combines regex filters, PII checkers, and external validation models 7 | Moderate; requires wrapping model calls with a Guard 7 | Competitor mentions, PII leakage, and off-topic dialogue 6 |
| **NVIDIA NeMo Guardrails** 11 | \< 5 ms 11 | Uses Colang script definitions to enforce dialogue flows and safety boundaries 11 | High; requires defining Colang files and orchestrating flow layers 11 | Jailbreak attempts, prompt injections, and off-topic conversations 7 |

## **State Modification, Time Travel, and Human-in-the-Loop Orchestration**

When automated self-correction loops fail to resolve behavioral anomalies, the system must support manual interventions and state corrections.1 This capability relies on robust state persistence.1 Swapping checkpointers allows developers to transition the application from local development to production scale without modifying the underlying graph structure.5

* **InMemorySaver**: Keeps state transitions in local memory, making it ideal for rapid debugging and unit testing.5  
* **SqliteSaver**: Persists state transitions to a local SQLite database, providing light durability for single-server deployments.5  
* **PostgresSaver**: Connects to a PostgreSQL instance, offering scalable, multi-instance persistence for production environments.5  
* **DynamoDBSaver**: Integrates with Amazon DynamoDB, using the thread ID as the partition key (PK) and the checkpoint ID as the sort key (SK) for highly scalable state storage.20

By persisting state transitions, the system can retrieve the execution history of any thread in reverse chronological order using the get\_state\_history API.19 If an agent makes an incorrect decision, the operator can rewind execution to a prior checkpoint, modify the state variables using the update\_state API, and resume execution from that point.1  
This creates a new fork in the thread's execution history, preserving the original timeline while allowing the agent to proceed down an corrected path.1

                     \+-------------------------+  
                     |  Checkpoint 1: Initial  |  
                     \+------------+------------+  
                                  |  
                                  v  
                     \+-------------------------+  
                     | Checkpoint 2: Error     |  
                     \+------------+------------+  
                                  |  
                                  \+-----------------------+  
                                  | (Fork Timeline)       |  
                                  v                       v  
                     \+-------------------------+ \+-------------------------+  
                     | Checkpoint 3 (Original):| | Checkpoint 3 (Corrected):|  
                     | Failed Execution        | | Edited State Values     |  
                     \+-------------------------+ \+------------+------------+  
                                                              |  
                                                              v  
                                                 \+-------------------------+  
                                                 | Checkpoint 4: Resumed   |  
                                                 | Correct Path            |  
                                                 \+-------------------------+

| Time Travel Step | API Execution | Core Mechanism | Intended Outcome |
| :---- | :---- | :---- | :---- |
| **1\. Identify Failure** | get\_state\_history(thread\_config) 19 | Queries the database backend to retrieve all past state snapshots for a specific thread.5 | Locates the precise checkpoint prior to the agent's failure.1 |
| **2\. Edit State Values** | update\_state(config, values, as\_node) 19 | Injects corrected values into the state, attributing the change to a specific node.19 | Modifies incorrect state variables to align the workflow with expected results.1 |
| **3\. Resume Execution** | invoke(None, fork\_config) 19 | Resumes processing from the newly edited checkpoint without re-running prior steps.1 | Executes the corrected workflow path, avoiding previous mistakes.1 |

When running asynchronous streams using astream with interrupt(), developers must be careful about how state changes are persisted.22 If a node appends a value to a state list and immediately triggers an interrupt() in the same step, the system may pause before the state update is fully saved to the checkpointer.22  
To prevent this issue, developers should split the workflow into two consecutive nodes.22 The first node updates the state and returns the updated values to the graph, and the second node immediately calls the interrupt() function to pause execution.22  
Additionally, if the application contains complex, nested workflows, developers can enable checkpointers on individual subgraphs (checkpointer=True).19 This isolates the state history of each subgraph, allowing developers to perform localized time travel and state recovery within a nested workflow without disrupting the master graph's execution history.19

## **Evaluation and Alignment via Post-Training Methodologies**

While runtime safety filters and state recovery loops prevent errors from reaching users, they add latency and increase API costs.2 To permanently fix behavioral anomalies, developers must align the base model's weights using targeted post-training methodologies.23  
The five educational courses from DeepLearning.AI offer different tools and frameworks for addressing these validation and alignment challenges.8

### **Safe and Reliable AI via Guardrails**

This course provides immediate, runtime safety and validation strategies.8 It explains how to build input and output validation gates using pre-built validators from the Guardrails Hub, focusing on NLI-based hallucination checks, PII filtering, and competitor detection.6  
While highly effective for blocking incorrect outputs at runtime, these techniques act as external filters and do not modify the underlying weights of the model.7

### **Post-training of LLMs**

This course covers the key techniques used to adapt base language models into instruction-following assistants, focusing on Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Online Reinforcement Learning (RL).24  
For behavioral corrections, such as correcting incorrect responses to user gestures, DPO is highly effective.24 DPO optimizes the model's weights using pairwise preference datasets containing a prompt, a preferred response (![][image1]), and a dispreferred response (![][image2]).24 The training objective is mathematically formulated as:  
![][image3]  
This objective updates the policy ![][image4] relative to a reference model ![][image5], increasing the likelihood of generating the preferred response while penalizing the incorrect output.24 This allows developers to permanently correct dialogue errors without the complexity of training an auxiliary reward model.24

### **Reinforcement Fine-Tuning LLMs with GRPO**

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that optimizes model outputs using rule-based reward functions.24 GRPO is particularly useful for tasks with verifiable outcomes, such as enforcing structural formats or strict color list sizes.25  
Unlike traditional RL algorithms that require a separate critic model, GRPO generates a group of ![][image6] outputs for a single prompt and calculates their relative advantages.25 The GRPO loss function is defined as:  
![][image7]  
The advantage ![][image8] is calculated by normalizing the rewards across the group:  
![][image9]  
where ![][image10] and ![][image11] represent the mean and standard deviation of the rewards within the sampled group.25 By designing a reward function that assigns high scores to outputs that contain exactly two colors and penalizing single-color or incorrect outputs, GRPO trains the model to consistently adhere to structural constraints and factual requirements.25

### **Reinforcement Learning from Human Feedback (RLHF)**

This course explains the classical reinforcement learning framework used to align model behaviors with human preferences.27 It details how to collect preference datasets, train reward models, and run optimization loops using Proximal Policy Optimization (PPO).27  
While RLHF is foundational to model alignment, its multi-model architecture (incorporating an actor, critic, reference, and reward model) introduces high engineering overhead.24  
For localized behavioral corrections, direct optimization techniques like DPO or GRPO are generally more efficient and easier to implement.24

### **Pretraining LLMs**

This course focuses on the initial stage of LLM training, where a base model is trained on unlabeled text to predict the next token.32  
Because pre-training only establishes general language representations and does not train the model to follow instructions or conform to conversational structures, it is not suitable for fixing localized dialogue errors or runtime behavioral anomalies.30

| DeepLearning.AI Course | Target Alignment Phase | Primary Practical Application | Direct Relevance to Error Correction |
| :---- | :---- | :---- | :---- |
| **Safe and Reliable AI via Guardrails** 8 | Runtime Validation Layer 8 | Intercepts malformed inputs and output hallucinations before they reach downstream components.6 | **High**: Best for immediate, non-intrusive safety checks and semantic filters.7 |
| **Post-training of LLMs** 24 | Offline Model Alignment 24 | Uses pairwise preference datasets and DPO to correct persistent dialogue style and conversational errors.24 | **High**: Excellent for correcting conversational and style failures.24 |
| **Reinforcement Fine-Tuning LLMs with GRPO** 25 | Offline Reasoning & Formatting Optimization 24 | Trains models on structured constraints and formatting rules using automated, rule-based reward functions.25 | **High**: Best for enforcing strict schema rules and cardinality requirements.25 |
| **Reinforcement Learning from Human Feedback** 27 | Systemic Human-Preference Alignment 27 | Standardizes general safety, helpfulness, and conversational capabilities across broad domains.27 | **Moderate**: Highly valuable for general alignment but introduces significant architectural complexity.24 |
| **Pretraining LLMs** 26 | Base Language Modeling 32 | Builds base models or specializes models for specific domains using large-scale text corporas.32 | **Low**: Intended for base model creation; cannot resolve specific, post-deployment agent failures.30 |

## **Synthesis and Engineering Recommendations**

To build a stateful agentic system that minimizes behavioral anomalies, false positives, and false negatives, developers should implement a tiered, defense-in-depth architecture.

\+---------------------------------------------------------------------------------+  
|                                 TIER 1: RUNTIME                                 |  
|                         (LangGraph & Cyclic PEV Loops)                          |  
|                                                                                 |  
|  \* Isolate planning and validation nodes.                         |  
|  \* Implement deterministic, non-LLM router nodes for recovery.          |  
|  \* Use persistent checkpointers (e.g., PostgresSaver, DynamoDBSaver).   |  
|  \* Configure interrupts for manual state updates via time travel.  |  
\+------------------------------------+--------------------------------------------+  
                                     |  
                                     v  
\+---------------------------------------------------------------------------------+  
|                                TIER 2: VALIDATION                               |  
|                         (Active Input/Output Guardrails)                        |  
|                                                                                 |  
|  \* Enforce strict schemas using Pydantic validation loops.       |  
|  \* Deploy semantic NLI checks to verify dialog tree transitions.  |  
|  \* Format output dictionary structures to prevent serialization errors. |  
\+------------------------------------+--------------------------------------------+  
                                     |  
                                     v  
\+---------------------------------------------------------------------------------+  
|                                TIER 3: ALIGNMENT                                |  
|                        (Offline Post-Training Iteration)                        |  
|                                                                                 |  
|  \* Collect validation logs and failure traces from state databases.     |  
|  \* Use DPO to correct conversational errors and gesture mappings.|  
|  \* Apply GRPO with reward metrics to reinforce formatting rules. |  
\+---------------------------------------------------------------------------------+

### **1\. Tiered Runtime Validation**

The first line of defense should be built directly into the graph architecture.2 Developers should structure conversational agents using the Plan-Execute-Validate (PEV) pattern, keeping LLM-based planning and structured validation in separate nodes.2  
If validation fails, a deterministic router node should intercept the error, update the graph's state with the failure feedback, and redirect execution back to the planner.2  
To handle unrecoverable errors, the graph should use persistent checkpointers.5 This allows human operators to pause execution, inspect the thread's history, modify incorrect state variables, and resume processing from a safe fork.1

### **2\. Guardrails and Schema Enforcement**

To prevent structural and factual errors, developers should enforce Pydantic schemas using the model's structured output interface.12 Any schema parsing failures should be caught by the validator node and fed back into localized retry loops, preventing malformed payloads from disrupting the master graph.2  
For conversational dialogue mapping, developers should use NLI-based semantic guardrails.6 These guardrails verify that generated responses logically follow user inputs, allowing the system to substitute correct dialogue options or trigger retries when a semantic deviation is detected.7

### **3\. Continuous Post-Training Optimization**

To reduce the system's reliance on runtime validation gates, developers should implement a continuous post-training pipeline.24 By collecting validation logs and failure traces from the state database, engineers can compile high-quality alignment datasets.28  
For dialogue, style, and semantic errors, developers should run offline DPO training runs to optimize the model's conversational policy.24 For structural, mathematical, and formatting constraints, developers should use GRPO with automated validation scripts to train the model to consistently adhere to schema requirements.25  
This combination of runtime validation, checkpoint-driven recovery, and continuous model alignment ensures the system remains reliable, compliant, and cost-effective.2

#### **Works cited**

1. Human-in-the-Loop AI: Time-Travel Workflows with LangGraph \- Christian Mendieta, accessed June 4, 2026, [https://christianmendieta.ca/human-in-the-loop-ai-time-travel-workflows-with-langgraph/](https://christianmendieta.ca/human-in-the-loop-ai-time-travel-workflows-with-langgraph/)  
2. Building a Reliable LangGraph Workflow: Plan-Execute-Validate (PEV), Automated Retries, and MCP Integration in One Template \- DEV Community, accessed June 4, 2026, [https://dev.to/manjunathgovindaraju/building-a-reliable-langgraph-workflow-plan-execute-validate-pev-automated-retries-and-mcp-1pik](https://dev.to/manjunathgovindaraju/building-a-reliable-langgraph-workflow-plan-execute-validate-pev-automated-retries-and-mcp-1pik)  
3. How AI Systems Self-Correct with LangGraph? \- Hexmos Journal, accessed June 4, 2026, [https://journal.hexmos.com/how-ai-systems-self-correct-with-langgraph/](https://journal.hexmos.com/how-ai-systems-self-correct-with-langgraph/)  
4. LangGraph Tutorial: Self-Correcting AI Agents and Agent Loops | ActiveWizards, accessed June 4, 2026, [https://activewizards.com/blog/a-deep-dive-into-langgraph-for-self-correcting-ai-agents/](https://activewizards.com/blog/a-deep-dive-into-langgraph-for-self-correcting-ai-agents/)  
5. LangGraph in Production: Building Stateful AI Agents \- Kalvium Labs, accessed June 4, 2026, [https://www.kalviumlabs.ai/blog/langgraph-in-production-stateful-multi-step-agents/](https://www.kalviumlabs.ai/blog/langgraph-in-production-stateful-multi-step-agents/)  
6. Safe-and-reliable-AI-via-guardrails/README.md at main \- GitHub, accessed June 4, 2026, [https://github.com/sdivyanshu90/Safe-and-reliable-AI-via-guardrails/blob/main/README.md](https://github.com/sdivyanshu90/Safe-and-reliable-AI-via-guardrails/blob/main/README.md)  
7. What Are AI Guardrails \- Last9, accessed June 4, 2026, [https://last9.io/blog/what-are-ai-guardrails/](https://last9.io/blog/what-are-ai-guardrails/)  
8. Safe and reliable AI via guardrails \- DeepLearning.AI, accessed June 4, 2026, [https://www.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails](https://www.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails)  
9. Safe and reliable AI via guardrails \- DeepLearning.AI \- Learning Platform, accessed June 4, 2026, [https://learn.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails/lesson/rz5a6/introduction](https://learn.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails/lesson/rz5a6/introduction)  
10. Safe and reliable AI via guardrails \- DeepLearning.AI \- Learning Platform, accessed June 4, 2026, [https://learn.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails/lesson/ph1aa/building-your-first-guardrail](https://learn.deeplearning.ai/courses/safe-and-reliable-ai-via-guardrails/lesson/ph1aa/building-your-first-guardrail)  
11. Best Guardrails Tools for AI Agents in 2026 \- Fast.io, accessed June 4, 2026, [https://fast.io/resources/best-guardrails-tools-ai-agents/](https://fast.io/resources/best-guardrails-tools-ai-agents/)  
12. Structured output \- Docs by LangChain, accessed June 4, 2026, [https://docs.langchain.com/oss/python/langchain/structured-output](https://docs.langchain.com/oss/python/langchain/structured-output)  
13. Mastering Pydantic for LLM Workflows | by DhanushKumar | Artificial Intelligence in Plain English, accessed June 4, 2026, [https://ai.plainenglish.io/mastering-pydantic-for-llm-workflows-c6ed18fc79cc](https://ai.plainenglish.io/mastering-pydantic-for-llm-workflows-c6ed18fc79cc)  
14. Parsing error with structured output with model output \- LangChain Forum, accessed June 4, 2026, [https://forum.langchain.com/t/parsing-error-with-structured-output-with-model-output/2985](https://forum.langchain.com/t/parsing-error-with-structured-output-with-model-output/2985)  
15. How to retry and fix with\_structured\_output parsing error : r/LangChain \- Reddit, accessed June 4, 2026, [https://www.reddit.com/r/LangChain/comments/1nr0duf/how\_to\_retry\_and\_fix\_with\_structured\_output/](https://www.reddit.com/r/LangChain/comments/1nr0duf/how_to_retry_and_fix_with_structured_output/)  
16. LangChain \- Guardrails AI, accessed June 4, 2026, [https://guardrailsai.com/guardrails/docs/integrations/langchain](https://guardrailsai.com/guardrails/docs/integrations/langchain)  
17. 5 Best Tools to Implement Guardrails for AI Applications, accessed June 4, 2026, [https://www.getmaxim.ai/articles/5-best-tools-to-implement-guardrails-for-ai-applications/](https://www.getmaxim.ai/articles/5-best-tools-to-implement-guardrails-for-ai-applications/)  
18. Human in the loop (HITL) AI Agents with LangGraph & Elastic \- Elasticsearch Labs, accessed June 4, 2026, [https://www.elastic.co/search-labs/blog/human-in-the-loop-hitllanggraph-elasticsearch](https://www.elastic.co/search-labs/blog/human-in-the-loop-hitllanggraph-elasticsearch)  
19. Use time-travel \- Docs by LangChain, accessed June 4, 2026, [https://docs.langchain.com/oss/python/langgraph/use-time-travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)  
20. The Missing Piece in Your LangGraph Workflow | by OverTheHead | AWS in Plain English, accessed June 4, 2026, [https://aws.plainenglish.io/the-missing-piece-in-your-langgraph-workflow-a5c390ed2af4](https://aws.plainenglish.io/the-missing-piece-in-your-langgraph-workflow-a5c390ed2af4)  
21. Time travel using the server API \- Docs by LangChain, accessed June 4, 2026, [https://docs.langchain.com/langsmith/human-in-the-loop-time-travel](https://docs.langchain.com/langsmith/human-in-the-loop-time-travel)  
22. Update state messages list before interrupt \- Stack Overflow, accessed June 4, 2026, [https://stackoverflow.com/questions/79866929/update-state-messages-list-before-interrupt](https://stackoverflow.com/questions/79866929/update-state-messages-list-before-interrupt)  
23. Finetuning Large Language Models \- DeepLearning.AI, accessed June 4, 2026, [https://www.deeplearning.ai/courses/finetuning-large-language-models](https://www.deeplearning.ai/courses/finetuning-large-language-models)  
24. Post-training of LLMs \- DeepLearning.AI, accessed June 4, 2026, [https://www.deeplearning.ai/courses/post-training-of-llms](https://www.deeplearning.ai/courses/post-training-of-llms)  
25. Reinforcement Fine-Tuning LLMs With GRPO \- DeepLearning.AI, accessed June 4, 2026, [https://www.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo](https://www.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo)  
26. Pretraining LLMs \- DeepLearning.AI \- Learning Platform, accessed June 4, 2026, [https://learn.deeplearning.ai/courses/pretraining-llms/course-feedback](https://learn.deeplearning.ai/courses/pretraining-llms/course-feedback)  
27. Reinforcement Learning From Human Feedback \- DeepLearning.AI, accessed June 4, 2026, [https://www.deeplearning.ai/courses/reinforcement-learning-from-human-feedback](https://www.deeplearning.ai/courses/reinforcement-learning-from-human-feedback)  
28. Review of LLM Post-Training Techniques | by Sulbha Jain \- Medium, accessed June 4, 2026, [https://sulbhajain.medium.com/review-of-llm-post-training-techniques-25c2e049954e](https://sulbhajain.medium.com/review-of-llm-post-training-techniques-25c2e049954e)  
29. Reinforcement Learning from Human Feedback \- arXiv, accessed June 4, 2026, [https://arxiv.org/html/2504.12501v3](https://arxiv.org/html/2504.12501v3)  
30. Reinforcement Learning From Human Feedback \- DeepLearning.AI, accessed June 4, 2026, [https://learn.deeplearning.ai/courses/reinforcement-learning-from-human-feedback/lesson/k1m9r/introduction](https://learn.deeplearning.ai/courses/reinforcement-learning-from-human-feedback/lesson/k1m9r/introduction)  
31. RLHF: Understanding Reinforcement Learning from Human Feedback \- Coursera, accessed June 4, 2026, [https://www.coursera.org/articles/rlhf](https://www.coursera.org/articles/rlhf)  
32. Pretraining LLMs \- DeepLearning.AI \- Learning Platform, accessed June 4, 2026, [https://learn.deeplearning.ai/courses/pretraining-llms/lesson/z2ntd/introduction](https://learn.deeplearning.ai/courses/pretraining-llms/lesson/z2ntd/introduction)  
33. Pretraining LLMs \- DeepLearning.AI, accessed June 4, 2026, [https://www.deeplearning.ai/alpha/short-courses/pretraining-llms](https://www.deeplearning.ai/alpha/short-courses/pretraining-llms)  
34. Post-training of LLMs \- DeepLearning.AI \- Learning Platform, accessed June 4, 2026, [https://learn.deeplearning.ai/courses/post-training-of-llms/lesson/ynmgf/introduction-to-post-training](https://learn.deeplearning.ai/courses/post-training-of-llms/lesson/ynmgf/introduction-to-post-training)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABRUlEQVR4Xu3TzStEYRTH8YNElLeFSeQlWwtlZeNlMV7iT7CSbEQiG0qzIln4KxSlyVuSslKykqRkoSysLISFjbx8n84xnekuzGTB4v7q09xzzu3e+zx3rkicOHH+Lv3Yww2GXL8W90i6Xs5pxC4KcYFtNxvFJ9pcL+dMogvN+MCcm23gEQWul3dSeEPC6nCxB9HV/CrXOHJ1u+g2zLpe3ikRvciC681Yr8PqeRyj1+oUxtGCO1RaPytFonu5ZHUDbvEk+lIH0INNjNk55+hGMa5QZ/1IhnGJfWzhHTs2q0GZ6I3CO6jAK0ptvma/kTSh3tV9otsQlvqdQZzacZifudmyO86kCi84cb0D0Q+j3PVGkLbjadGVhYQbhi2JJOxnWPaE1VN4RmfmDE01DrEi+iLDh7QqP/xrFkVPDE+xjtbscZz/mC/BqznQMLw7hQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABIUlEQVR4Xu3Sq0sEURTH8aOrLrLFoiKKClaDIAgabD5Ai7DJuGHLsgbFJIjJatE/QTAYfCEo2wSzCIsYbCaD+Gri43s55w5n2MVsmB98YH/n3t25MzsiWbL8p8zhFPdYcPMePGLGzRoyiBO04gZHbq2EH4y6WUOqmMYwvrHu1g7wjBbMo45dt57KFj7Raz186Un0dDEXWHY9lTtcuj4mevw16+14R1+ywyUvunnDzVZtNm59SvQiTZMTvddt6wN4wIvoww0JP75nn5tmEbc4wyG+cOzWayi6nsoQ+l2fFT1+2XoHPtCNzrgppgtvuHKzc9EXqGB9QvQvDM9hKW6KCfcbjluxvoJXTCY79MrX2EGbmyfZFH0Dw/3vYyS9nOWv/AISyzJeVSn3KgAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,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EwMiIiLTkrCxoepKL3g5srjulTaU+wbyBs1KUF/Zuk++dJdYb7NqRzLKV7OxItK3vfDtIuiBwt75sq2OpeM1Z2k6Nm7fQhWpKZmYBZwL3weSc+FVM/i5YBqya6lCyzGn0rhnvs/Fc7GDpWSPCheJH3xhUJ35GMg4fxz/Yy0d+/g84Ng3rW5fLDnvxICIiMi0JGybWf/mytOk3DeQN2KfqkNceMLikHIribLC87dwfUPrfWXTYcV940ASwkbPq5Gfix37oum4x4StPBebW+97mTmPe1nay4/94Jqm5OdjoCWOfdPU92KrdzsxICIiMi0Jm2PLk2lT7hu4saUkgL0IWZ1Jn9Ih+T53SnH94nzJ17SV1R6qRPwbVOu4bEoYloqEoc0+fLOGFZ5+LuB7OLJ6lApm/Mq98lxQVdvPUp8b5wMkayxS8CntaD4GWmBPzLpjz1RomypyJwZERET42iAqWwxWNK40epHoEZt1fJXZSmH/PukZ5VwcHAMtxKqrY5qXpK2J/x1Oa6VZREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREZH3yf1Z6hUkGn6YhAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAABH0lEQVR4Xu3SPSiFcRTH8ZNBSoyXGEzMujObGE1sLBYDwx0UJi8ZlFKUQRQpCTOlhNliYvOSjTIq8vY9/uc+9/yfwe2apOdXnzrnf56XOs8jkiXLP04OB3jFOz7w6Wh/lFxdJtU4xw6mcYdZLOISU6YrXF4+BYxYXYsLq7uxbHU6eaxhG82pWZR+rFs9jkk3K6YJVxJePoCZeBxnD4NWb2CsNEqyJWE1mj7su1kU/TAvaLH+GPOl8Xca8YZ260flhwdO4Nr1Jzh0vWYYD65fwpzrk1ThBqvubBfPqHFnm7iXsA71iF43T9KAJ3S4syHcos6dnUr4WJriPf6FFUfX0GO17m/BzX6VFXRK+GXOUB+PK0+rhJ9Zd92WmmX5q/kCWDg11hwpS8IAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAaCAYAAABLlle3AAABY0lEQVR4Xu3UvyvFURjH8cePpBBKSEkGIYs/wEJSJiajlJHFQFiQBQMDUqKklDDIgJT8AQYmWfwqVhshv95P53G/xzfR1dfk+6lX9zznfG/n3nOee0XixIkT55sUYhtPeMEr3jxa7yWejiAZOMQqRnCFUUzhBMOm0T0eTXrQbeMsHNu4CdM2/tO0YdHG/Rj01n6TVszhDOWhtUTW0W7jJfQGS0knF5dIxyxKP61atJkeUWb1PsaD5aRTidPwZDgDOPfqA+x4tWYT9+JOQJusXlwjTmICK2hAtrhTu8MCavTN4aTiAvPe3Jq4DTK9uTRxP6lmdKACQxI0XAlubFyFaxt/mSLcos6b6xR3JznenOYZxV6t3b4lwU9Lr0W/6Y+bJhPdNN+rj9Dl1R+plgg31ePN8+o+7CLFar1vvefINl0W97e4gVqb034YE9dk+toi7vj1mQfMoMCe/dPoB9Gmi/NP8g6TcESf2aE56gAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAAA/klEQVR4Xu2SvWoCURCFj+IPNop9AnkDG7GzsEglVr6CbZLCNPZiZ5NAIM/gC4hgkUpUrAQrCxslEAQ7URA949zoOipYWuwHHyxz7t6de3cAn/slSV/pD53QER3Qksvr9Nk9nyGL5rRNszTo6mH6STt0SWOufiBAv+mWvpnsH9lkRps2EGrQl6s2MLToiy1m6IZOcaE1wxd9ssUG9OsVG9zKArpB2ga3EIG+vMbxxr3ImYfQy/ulY/p+soL8Qe8gZAMPE7qiCVPf8wHtomADRwqaSzcXidMetJMi9H8LcqQc7UKH59p87InSMnRkZRLlrH3o2D7QPH08rPbx8bADCRAvXUD5FDUAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAbCAYAAACTHcTmAAABFUlEQVR4XmNgGAWjgBDwB+L36IKUACEgvgHE/4GYF02ObBAKxPMYIIaqocmRBRyBWA6I2xkghtqjSpMO+IE4GsouZoAYGomQJg8UADELlB3HADG0CCFNOnAFYiskvjcDxNBuJDEYMAPi2+iC2ADVDeUD4nNAfAEJ32SAGLoUSR0MCAOxHbogOuhigChEBjIMEEP3o4kTBTyAuBBdEAjYGSCGglwMA4xAXA/EuxjwuNQQiJ8wYM81IAO+QjEMuDNADFsOxMlI4mBgDsSPGCAuAeFnQCyAJD8BiJ8jyT8E4haoGpAD3jFAsjIKALmCGV2QSBAPxOsYIBkFlqYpBtuAOBiIK4GYDU2ObNABxAsYIOl4FAw1AAAIWTIT7leF4wAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA8CAYAAADbhOb7AAACYklEQVR4Xu3dy6tNURwH8OURJfJKMTGQUjIWM6EY8BcoAxMTdAeYGsgfYCiTSykjI+RRlDyjKEoxU8pASMrI47fa6zrr7O6De0/tc/L51Lfz+62160x/rXP23ikBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADA0JofeVf1FyM/qx4AgI6diOyt+huRX1U/KOer+mhVAwAwg1uRxaVenZph7UJveyAWRr5V/eOqBgBgGrtT7zRtSeRRaoarCWOR8aqfrZOp9z07qhoAgBk8j3yp+puRXVU/KJ9Tb/DL3/Eqsv3PLgAAU8o3F1wrdT5Z+x7ZXPpVkQep+Zl0rvKJ2ulSf03N/9nu9LYBAJjMm9QMUh8iK8vavsjDyPWy9qysz9XH1PzcejdyKPI2srTvCgAAZuVYGswJ2+X2AgDAqMsnX/vbix04G9nQXvxHByNb2osAAKNs4s7NU+0NAACGRx7YzrUXJ7En8mKaLO9dCgDAXK2NHC51HtieVntbqxoAgI5cirxMzcnYj8j7am8Qf/yv5YFwGAIAMDLWtfr86qaJgWZeal4fBQBABxZFXqdmONtZ1g6UPicPaisin8oeAAAdqN/dOZUr6e+uAwCgI+PtBQAAAABgRG2L3E79d3su6LsCAIDObIocL/WyyFi1BwDAEMjPiptf9WeqGgCAjq1P/Q+7PRLZWPUAAAyBq+UzP2LkXr0BAMBwWBO5H3mSmof+AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPy/fgM/r3G0YyOcZgAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAA1ElEQVR4XmNgGAXDAswG4gVoYrlAfBJNDAxYgPgLEE9EEz8BxNvRxMDAHIj/A3EokhgXEP8C4nokMTgoY4BokEASc4CKuSGJwQHJGrYA8W00sVog/gfE/GjiDExA/J4BM4R2AvEVKNsSWUKfAWJ1M5IYyNRPDJCgZgbifUhyDDkMEA1roXweIF4PxD+BuAmIfYG4AioHBquA+A0QH2eARNJ+IHYG4iQGiL+2MkAMgYNnQLwCWQAfUGaAOCcbXQIXiGOAaNBBl8AFCoD4DBAzoksMVwAAhOgqRM26O90AAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAAoUlEQVR4XmNgGAXDAsQC8UUg/gfE/9HwNiR1YDABiH8A8XQgrgPih0D8C4gLgbgAiK0QSiEmgySRBb0YICYbIImBARMQPwLiRWjipgwQDa5o4qRrsIRK+KGJx0PFtdHEGeKgEhJo4msYIB5nRBNn8GGAaGBHElME4u8MEMMwAA8QP2dAOEkUiE8AcR9cBRZgBMSHgPgwEB8E4ihU6VEw2AAA+DAj0uiWDnAAAAAASUVORK5CYII=>