I want to send multiple urls and a mixture of multiple urls, and documents (local files) to the update_avatar_identity_with_media endpoint and
  have everything processed (each individual upload in parallel);
    I need to be able to send a playlist (youtube) and have all the items processed.
  There may be files (text and markdown) that contains a list of items to process (urls) and each should be processed accordingly. Example of 
  file with multiple urls: /home/user/gh/anubis-project/wt/f-multi-file-uploads/data/shivon_zilis/confirmed_search_results_list.txt
  (for playlists I need to skip content that is already uploaded [identified from /list_avatar_media function logic] because there are hundreds of files; removal of media uses the /delete_avatar_media endpoint function logic)
  (could be .md)
  
  
  Example of playlist:
  Gracie Abrams
https://www.youtube.com/playlist?list=PL9rU625vkl4XLqULoLJkid_iOxgtQnn8O
  
  Shivon Zilis
https://www.youtube.com/playlist?list=PL9rU625vkl4XSOQDZxdFhVZgows3FJiKH
  
  Elon Musk
https://www.youtube.com/playlist?list=PL9rU625vkl4UWrbm5S1ZRRU4bk-eYbHr9
  
  Y Combinator Startup
https://www.youtube.com/watch?v=EN7frwQIbKc&list=PLQ-uHSnFig5M9fW16o2l35jrfdsxGknNB

















{"job_id":"10259147-adcf-416e-84dc-69349dc75bc7","status":"queued","progress_url":"/media_job/10259147-adcf-416e-84dc-69349dc75bc7/progress","cancel_url":"/media_job/10259147-adcf-416e-84dc-69349dc75bc7/cancel","items_accepted":12,"filenames":["https://www.youtube.com/watch?v=yPIcDfKqg1g","https://www.youtube.com/watch?v=WV329HQvzUw","https://www.youtube.com/watch?v=iRkPMqwX5b8","https://www.youtube.com/watch?v=W3UtZOIbDjY&t=1378s","https://www.youtube.com/watch?v=gIF_D6iUusU","https://www.youtube.com/watch?v=-tQwzhHjAVI","https://www.youtube.com/watch?v=a-yR9W2EV7w","https://www.youtube.com/shorts/NHbZ1sfj4dM","https://lifeboat.com/ex/bios.shivon.a.zilis","https://x.com/shivon","https://grokipedia.com/page/Shivon_Zilis","https://www.instagram.com/shivonzilis_official/?hl=en"],"items":[{"job_id":"40d91500-9f38-4626-a09a-7e641941c6d9","filename":"https://www.youtube.com/watch?v=yPIcDfKqg1g","status":"queued","progress_url":"/media_job/40d91500-9f38-4626-a09a-7e641941c6d9/progress","cancel_url":"/media_job/40d91500-9f38-4626-a09a-7e641941c6d9/cancel"},{"job_id":"e6c27c73-cee7-4d5d-b5ec-e8cfa1824747","filename":"https://www.youtube.com/watch?v=WV329HQvzUw","status":"queued","progress_url":"/media_job/e6c27c73-cee7-4d5d-b5ec-e8cfa1824747/progress","cancel_url":"/media_job/e6c27c73-cee7-4d5d-b5ec-e8cfa1824747/cancel"},{"job_id":"ac3c9206-8363-4fad-96b0-e37c6e01a3c7","filename":"https://www.youtube.com/watch?v=iRkPMqwX5b8","status":"queued","progress_url":"/media_job/ac3c9206-8363-4fad-96b0-e37c6e01a3c7/progress","cancel_url":"/media_job/ac3c9206-8363-4fad-96b0-e37c6e01a3c7/cancel"},{"job_id":"94494fb5-8c15-49ff-9f2e-2812009877f8","filename":"https://www.youtube.com/watch?v=W3UtZOIbDjY&t=1378s","status":"queued","progress_url":"/media_job/94494fb5-8c15-49ff-9f2e-2812009877f8/progress","cancel_url":"/media_job/94494fb5-8c15-49ff-9f2e-2812009877f8/cancel"},{"job_id":"c8e13736-058f-4bc6-8883-e6345fc105dc","filename":"https://www.youtube.com/watch?v=gIF_D6iUusU","status":"queued","progress_url":"/media_job/c8e13736-058f-4bc6-8883-e6345fc105dc/progress","cancel_url":"/media_job/c8e13736-058f-4bc6-8883-e6345fc105dc/cancel"},{"job_id":"afd6a1ed-f891-4edf-bd75-a3ab2c233002","filename":"https://www.youtube.com/watch?v=-tQwzhHjAVI","status":"queued","progress_url":"/media_job/afd6a1ed-f891-4edf-bd75-a3ab2c233002/progress","cancel_url":"/media_job/afd6a1ed-f891-4edf-bd75-a3ab2c233002/cancel"},{"job_id":"33c89b0b-bbeb-4189-921c-1a28e7a66fb5","filename":"https://www.youtube.com/watch?v=a-yR9W2EV7w","status":"queued","progress_url":"/media_job/33c89b0b-bbeb-4189-921c-1a28e7a66fb5/progress","cancel_url":"/media_job/33c89b0b-bbeb-4189-921c-1a28e7a66fb5/cancel"},{"job_id":"cf659c99-e155-4a2f-9a34-6fb332713f47","filename":"https://www.youtube.com/shorts/NHbZ1sfj4dM","status":"queued","progress_url":"/media_job/cf659c99-e155-4a2f-9a34-6fb332713f47/progress","cancel_url":"/media_job/cf659c99-e155-4a2f-9a34-6fb332713f47/cancel"},{"job_id":"b7877ee9-403a-4c15-a41d-a2cbfc1aaf1e","filename":"https://lifeboat.com/ex/bios.shivon.a.zilis","status":"queued","progress_url":"/media_job/b7877ee9-403a-4c15-a41d-a2cbfc1aaf1e/progress","cancel_url":"/media_job/b7877ee9-403a-4c15-a41d-a2cbfc1aaf1e/cancel"},{"job_id":"7d90b4ae-db50-406d-9c14-5a7859fc2198","filename":"https://x.com/shivon","status":"queued","progress_url":"/media_job/7d90b4ae-db50-406d-9c14-5a7859fc2198/progress","cancel_url":"/media_job/7d90b4ae-db50-406d-9c14-5a7859fc2198/cancel"},{"job_id":"4754e14f-3e3e-4633-b0e2-22641c1afc8e","filename":"https://grokipedia.com/page/Shivon_Zilis","status":"queued","progress_url":"/media_job/4754e14f-3e3e-4633-b0e2-22641c1afc8e/progress","cancel_url":"/media_job/4754e14f-3e3e-4633-b0e2-22641c1afc8e/cancel"},{"job_id":"c423ee2f-a5af-489e-988b-8cf121de71c6","filename":"https://www.instagram.com/shivonzilis_official/?hl=en","status":"queued","progress_url":"/media_job/c423ee2f-a5af-489e-988b-8cf121de71c6/progress","cancel_url":"/media_job/c423ee2f-a5af-489e-988b-8cf121de71c6/cancel"}],"message":"Media processing started"}