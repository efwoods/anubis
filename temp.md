how do I open langsmith studio with a specific configuration? for example i want to implement the following route logic on a dedicated endpoint:


@app.post("/message/{assistant_id}")
async def message(
    request: Request,     
    assistant_id: str,    
    body: PublicAvatarMessagePayload,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
    ):

    logger.info("breakpoint")
    # allow for select avatar in query and anonymous user for a dedicated endpoint
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(detail="Error retrieving assistant information.", status_code=400)

    user_name = body.your_name
    user_description = body.your_description
    if request.headers.get("api-key") != '':

        user_id = current_user['identities'][0]['user_id']

        try:
            langgraph_client = get_client()
            assistant = await langgraph_client.assistants.get(assistant_id = assistant_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error selecting avatar.")
        
        config_update = {
                "configurable": {
                    "user_ctx": {"name":user_name, "description": user_description},
                    "user_id":user_id,
                    "assistant_id":assistant_id,
                    "assistant_ctx": {
                      "name": assistant.get("name", None),
                      "description": assistant.get("description", None),
                      "metadata": assistant.get("metadata", {})
                    }
                }
            }
    
    else:
        config_update = {
            "configurable": {
                "user_ctx": {"name":user_name, "description": user_description}
            }
        }
    # update with user information
    config['configurable'].update(config_update['configurable'])

        # store = app.state.store
    graph = app.state.response_only_graph

        # system_time = datetime.now(tz=timezone.utc).isoformat
        # content = [{"type":"text", "text": system_time}]
        # input = {"messages": HumanMessage(content=content)}
        # # store = make_pg_store()

        # result = await url_loading_graph.ainvoke(input, config=config)

        # logger.info(f"config: {config}")
        
    result = await graph.ainvoke(input={"messages":[HumanMessage(content=body.message)]}, config = config )

    logger.info(f"{result}")

    response = {}
    response["content"] = result['messages'][-1].content
    response_metadata = result['messages'][-1].response_metadata
    if response_metadata:
        response["response_metadata"] = response_metadata
    response['total_response_time_ms'] = ((time_ns() - start_time) // 1000000)
    return JSONResponse(response, status_code=200)

https://smith.langchain.com/o/68a934f8-fa1f-4950-a288-1bcedacf9f09/studio/thread?organizationId=68a934f8-fa1f-4950-a288-1bcedacf9f09&render=interact&baseUrl=http%3A%2F%2Flocalhost%3A8123&mode=chat&assistantId=d068614c-e072-41bc-8c45-b634c5e06a8c&threadId=681a4c36-cc77-421c-bc3e-98c88e1e6785

The url is unique to the assistant and any person using the url per assistant will be brought to langsmith studio with that above configuration. they will be logged in if they are using an api key. 


conversation threads need to be unique to the user_id and assistant_id and loaded if the user ID is not anonymous


langgraph.json
{
  "$schema": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/cli/langgraph/schemas/config.schema.json",
  "dependencies": ["."],
  "graphs": {
    "Anubis": "./src/anubis/graph.py:graph"
  },
  "store": {
    "index": {
      "dims": 384,
      "embed": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
      "fields": ["document.kwargs.page_content"]
    }
  },
  "env": ".env",
  "image_distro": "wolfi",
  "http": {
    "app": "./src/api/webapp.py:app"
  },
  "auth": {
    "path": "./src/security/auth.py:auth"
  }
}


<!-- response -->
<!-- 
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

router = APIRouter()

# Example: New GET endpoint to generate Studio URL, or integrate into your POST logic
@router.get("/studio/{assistant_id}")
async def open_studio(request: Request, assistant_id: str, current_user: dict = Depends(get_current_user_or_anonymous_user)):
    # Generate unique thread ID per user/access (resets TTL and starts fresh thread)
    user_id = current_user.get('identities', [{}])[0].get('user_id') if current_user.get("identities") else "anonymous"
    thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{assistant_id}:{int(request.url.make_absolute().hostname)}:{time.time_ns()}"))
    
    organization_id = "4442b4a8-8c3f-4761-b472-5b381af00ee1"
    base_url = str(request.url.scheme + "://" + request.url.hostname + ":" + str(request.url.port) + "/message/" + assistant_id)
    
    studio_url = (
        f"https://smith.langchain.com/o/{organization_id}/studio/thread?"
        f"organizationId={organization_id}&"
        f"render=interact&"
        f"baseUrl={base_url}&"
        f"mode=chat&"
        f"assistantId={assistant_id}&"
        f"threadId={thread_id}"
    )
    
    return RedirectResponse(studio_url) -->

https://smith.langchain.com/o/68a934f8-fa1f-4950-a288-1bcedacf9f09/studio/organizationID=68a934f8-fa1f-4950-a288-1bcedacf9f09&render=interact&baseURL=http://localhost:8123&mode=chat&assistantId=79fab19b-a868-480f-8f46-d31745d838b7
    