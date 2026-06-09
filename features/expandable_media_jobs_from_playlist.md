These urls are expanding and I want each item in the playlist to have an individual media job available immediately please


curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-QiYFReD7tLoK8fri6n9ZmYcSpqmJDw17mAD6OOKSNu8' \
  --form 'assistant_id=7f47c683-ae0c-4162-9a5a-42fbe8ba8fbb' \
  --form 'reference_audio=false' \
  --form 'reference_image=false' \
  --form 'all_speakers_target=false' \
  --form 'url=https://www.youtube.com/playlist?list=PL9rU625vkl4UlyAT5THtDV3cOB2KOkASX'


# incorrect: each item in the playlist should be expanded to an individual media job please
  "job_id": "878534de-919c-4ad6-8077-a24bf5e3a9f6",
  "status": "queued",
  "status_url": "/media_job/878534de-919c-4ad6-8077-a24bf5e3a9f6",
  "progress_url": "/media_job/878534de-919c-4ad6-8077-a24bf5e3a9f6/progress",
  "cancel_url": "/media_job/878534de-919c-4ad6-8077-a24bf5e3a9f6/cancel",
  "items_accepted": 0,
  "filenames": [],
  "items": [],
  "playlists_expanding": 1,
  "message": "Media processing started; enumerating 1 playlist(s) in the background"


# This is the current media job expanding each item; there should be a master job for all items and each sub item that is being processed from the update_avatar_identity_with_media endpoint should have its own job for progress indication and potential cancellation and to view if the item was skipped or an error occurred per item/job

data: {"type": "media_progress", "stage": "converting_child", "current": 1, "total": 1, "filename": "https://www.youtube.com/watch?v=NH5TOgftQBk", "item_job_id": "c681877b-a826-48dc-8685-c32d8da09a95", "item_filename": "test_playlist::Shivon Zilis: The Machine Intelligence Landscape", "started_at": 1780683509.2476425, "elapsed_seconds": 299.115}

data: {"type": "media_progress", "stage": "expanding", "url": "https://www.youtube.com/watch?v=CkUcCcRq_eM", "total": 1, "item_job_id": "862a031b-41f9-4c62-b773-9c887671f11d", "item_filename": "test_playlist::The Future of Brain Machine Interfaces - Shivon Zilis, Project Director at Neuralink | CUCAI 2021", "started_at": 1780683509.2476425, "elapsed_seconds": 299.115}

data: {"type": "media_progress", "stage": "converting_child", "current": 1, "total": 1, "filename": "https://www.youtube.com/watch?v=CkUcCcRq_eM", "item_job_id": "862a031b-41f9-4c62-b773-9c887671f11d", "item_filename": "test_playlist::The Future of Brain Machine Interfaces - Shivon Zilis, Project Director at Neuralink | CUCAI 2021", "started_at": 1780683509.2476425, "elapsed_seconds": 299.115}

data: {"type": "media_progress", "stage": "converting_complete", "total": 1, "skipped": 0, "errors": 0, "indexed": 15, "item_job_id": "e287219c-c02f-42c2-bc34-6c84d875a806", "item_filename": "test_playlist::Shivon Zilis Biography: The AI Expert Working Closely With Elon Musk", "started_at": 1780683509.2476425, "elapsed_seconds": 299.115}

data: {"type": "media_progress", "stage": "converting_complete", "total": 1, "skipped": 0, "errors": 0, "indexed": 11, "item_job_id": "9e7c5d46-bff8-4cf2-ae45-ded2ea68b3cd", "item_filename": "test_playlist::Shivon Zilis", "started_at": 1780683509.2476425, "elapsed_seconds": 299.115}

data: {"type": "media_progress", "stage": "indexing", "current": 1, "total": 1, "documents_indexed": 15, "documents_total": 15, "item_job_id": "e287219c-c02f-42c2-bc34-6c84d875a806", "item_filename": "test_playlist::Shivon Zilis Biography: The AI Expert Working Closely With Elon Musk", "started_at": 1780683509.2476425, "elapsed_seconds": 299.116}

data: {"type": "media_progress", "stage": "indexing", "current": 1, "total": 1, "documents_indexed": 10, "documents_total": 10, "item_job_id": "9e7c5d46-bff8-4cf2-ae45-ded2ea68b3cd", "item_filename": "test_playlist::Shivon Zilis", "started_at": 1780683509.2476425, "elapsed_seconds": 299.116}