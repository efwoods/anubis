# NATURAL LANGUAGE FACT EDITING:
The Harvard Business Analytics Program was completely separate from your Engineering Degree.







# Should be retry logic
/list_avatar_documents get_current_user needs to be retried on failutre
{
  "detail": "Authentication service temporarily unreachable."
}

# need a list media jobs endpoint 
/list_media_jobs




 openai.BadRequestError: Error code: 400 - {'error': {'message': '400: Part exceeded maximum size of 1024KB.', 'type': 'server_error', 'param': None, 'code': None}}

 langgraph-api-dev-1  | ERROR: unable to download video data: HTTP Error 403: Forbidden

  {"type": "media_progress", "stage": "item_error", "filename": "", "error": "process_text_to_document() missing 1 required positional argument: 'store'", "item_job_id": "a5f50dad-56dc-4645-835b-36a653e09190", "item_filename": "https://www.youtube.com/watch?v=gIF_D6iUusU", "started_at": 1781998880.3280032, "elapsed_seconds": 412.23}

langgraph-api-dev-1  | openai.BadRequestError: Error code: 400 - {'error': {'message': '400: Part exceeded maximum size of 1024KB.', 'type': 'server_error', 'param': None, 'code': None}}

{"type": "audio", "user_id": "69e5e49980b783d7dff3012b", "duration": null, "filename": "https://www.youtube.com/watch?v=gIF_D6iUusU", "namespace": "reference_audio", "created_at": "2026-06-21T00:04:25.789766+00:00", "assistant_id": "dbe60d13-89c5-4206-aa8d-8dd10592c559", "reference_audio": true, "adapter_acceptable": false, "namespace_filename": "01924d78-e306-583f-b077-92837a0b84c2", "processing_task_id": "f80b8f6b-b8ed-447b-bc71-2490185c134d", "analysis_acceptable": false, "vectorstore_acceptable": false}

langgraph-api-dev-1  |                                     ^^^^^^^^^^^^^^^^^^^^^^^^
langgraph-api-dev-1  |   File "/usr/lib/python3.11/site-packages/moviepy/audio/io/readers.py", line 235, in get_frame
langgraph-api-dev-1  |     self.buffer_around(fr_max)
langgraph-api-dev-1  |   File "/usr/lib/python3.11/site-packages/moviepy/audio/io/readers.py", line 281, in buffer_around
langgraph-api-dev-1  |     array = self.read_chunk(chunksize)
langgraph-api-dev-1  |             ^^^^^^^^^^^^^^^^^^^^^^^^^^
langgraph-api-dev-1  |   File "/usr/lib/python3.11/site-packages/moviepy/audio/io/readers.py", line 155, in read_chunk
langgraph-api-dev-1  |     s = self.proc.stdout.read(self.nchannels * chunksize * self.nbytes)
langgraph-api-dev-1  |         ^^^^^^^^^^^^^^^^
langgraph-api-dev-1  | AttributeError: 'NoneType' object has no attribute 'stdout'
langgraph-api-dev-1  | 2026-06-21T00:41:01.610434Z [warning  ] Error processing media: https://www.youtube.com/watch?v=CkUcCcRq_eM transcription_failed: 'NoneType' object has no attribute 'stdout' [src.subgraphs.process_media_graph.utils.nodes] api_revision=5206f65 api_variant=licensed langgraph_api_version=0.8.7 langgraph_node=convert_media_list_to_text_document thread_name=MainThread
langgraph-api-dev-1  | 2026-06-21T00:41:01.615134Z [info     ] analyze_documents: disabled via ENABLE_DOCUMENT_ANALYSIS; skipping [src.subgraphs.process_media_graph.utils.nodes] api_revision=5206f65 api_variant=licensed langgraph_api_version=0.8.7 langgraph_node=analyze_documents thread_name=MainThread


You know this day's really special in in some ways because it's this beautiful full circle moment. I spend pretty much a hundred percent of my time focusing on how human intelligence is gonna transform the world and hopefully transform the world for good and because of the decades of research here, a lot of the pioneering of this technology has been done in Canada, I get to bring a lot of my work home, which is just fantastic. And so we've been lucky enough to invest in two excellent machine intelligence companies based out of Toronto, and we can't wait to invest in more and and help

{"type": "audio", "user_id": "69e5e49980b783d7dff3012b", "duration": null, "filename": "https://www.youtube.com/watch?v=gIF_D6iUusU", "namespace": "reference_audio", "created_at": "2026-06-21T17:25:50.297558+00:00", "assistant_id": "dbe60d13-89c5-4206-aa8d-8dd10592c559", "reference_audio": true, "adapter_acceptable": false, "namespace_filename": "01924d78-e306-583f-b077-92837a0b84c2", "processing_task_id": "d6361f1b-4c8e-466c-bf54-6d6a2e8710df", "analysis_acceptable": false, "vectorstore_acceptable": false}


# Documents not batched (direct quotes)
 Batch 1/1: 183/183             [src.subgraphs.vector_store_graph.utils.helper_functions] api_revision=5206f65 api_variant=licensed langgraph_api_version=0.8.7 langgraph_node=index_docs thread_name=MainThread
langgraph-api-dev-1  | 2026-06-22T18:23:07.086980Z [info     ] breakpoint after batch_index_documents_vectorstore [src.subgraphs.vector_store_graph.index_graph] api_revision=5206f65 api_variant=licensed langgraph_api_version=0.8.7 langgraph_node=index_docs thread_name=MainThread
langgraph-api-dev-1  | 2026-06-22T18:23:07.093170Z [info     ] INDEXING DOCUMENTS             [src.subgraphs.vector_store_graph.index_graph] api_revision=5206f65 api_variant=licensed langgraph_api_version=0.8.7 langgraph_node=index_docs thread_name=MainThread
langgraph-api-dev-1  | 2026-06-22T18:23:07.093466Z [info     ] No documents to index; skipping batch indexing [src.subgraphs.vector_store_graph.index_graph] api_revision=5206f65 api_variant=licensed langgraph_api_version=0.8.7 langgraph_node=index_docs thread_name=MainThread
langgraph-api-dev-1  | 2026-06-22T18:23:07.100371Z [info     ] GET /media_job/3361943a-ad67-40f8-8a9d-e2333d853d68/progress 200 690760ms [langgraph_api.server] api_revision=5206f65 api_variant=licensed error_detail=None langgraph_api_version=0.8.7 latency_ms=690760 method=GET path=/media_job/3361943a-ad67-40f8-8a9d-e2333d853d68/progress path_params={'job_id': '3361943a-ad67-40f8-8a9d-e2333d853d68'} proto=1.1 query_string= req_header={} res_header={} response_size_b