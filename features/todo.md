need to tune tool choice
image with description model response format
token usage metrics
model does not respond initially


- aggregate model token use, cost, and latency
- integrate image description in messaging
- integrate extract personality from image
- integrate extract text from audio


/list_user_avatars: error:
langgraph-api-prod-1  | 2026-04-30T11:35:15.490441Z [info     ] breakpoint                     [user_router_module] api_revision=67d6599 api_variant=licensed langgraph_api_version=0.8.3 thread_name=MainThread
langgraph-api-prod-1  | 2026-04-30T11:35:15.492272Z [warning  ] discarding closed connection: <psycopg.AsyncConnection [BAD] at 0x7e4a5e2962d0> [psycopg.pool] api_revision=67d6599 api_variant=licensed langgraph_api_version=0.8.3 thread_name=MainThread
langgraph-api-prod-1  | 2026-04-30T11:35:15.494193Z [error    ] GET /list_user_avatars 500 4283ms [langgraph_api.server] api_revision=67d6599 api_variant=licensed error_detail=None langgraph_api_version=0.8.3 latency_ms=4283 method=GET path=/list_user_avatars path_params={} proto=1.1 query_string= req_header={} res_header={} response_size_bytes=90 route="APIRoute(path='/list_user_avatars', name='list_user_avatars', methods=['GET'])" run_id=None status=500 thread_name=MainThread ttfb_ms=4283.0
grafana-1             | logger=dashboard-service t=2026-04-30T11:35:26.796812049Z level=info msg="No last resource version found, starting from scratch" orgID=1
grafana-1             | logger=cleanup t=2026-04-30T11:35:26.840358057Z level=info msg="Completed cleanup jobs" duration=67.07824ms
grafana-1             | logger=plugins.update.checker t=2026-04-30T11:35:26.952819415Z level=info msg="Update check succeeded" duration=92.65832ms
grafana-1             | logger=plugins.update.checker t=2026-04-30T11:35:26.952879862Z level=info msg="flag evaluation succeeded" flag="{Value:false EvaluationDetails:{FlagKey:pluginsAutoUpdate FlagType:bool ResolutionDetail:{Variant:default Reason:STATIC ErrorCode: ErrorMessage: FlagMetadata:map[]}}}" details="{Value:false EvaluationDetails:{FlagKey:pluginsAutoUpdate FlagType:bool ResolutionDetail:{Variant:default Reason:STATIC ErrorCode: ErrorMessage: FlagMetadata:map[]}}}"


