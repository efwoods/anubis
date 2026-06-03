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


Dani Miranda Test Playlist Dataset
https://www.youtube.com/watch?v=KBnJLz1mifs&list=PL9rU625vkl4VU0EGvoTfkLue1shOqBZwV


# Dani Miranda Reference audio (video within the playlist)
https://youtu.be/KBnJLz1mifs?si=pxTSZhRAxVwY-w4q


curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-QiYFReD7tLoK8fri6n9ZmYcSpqmJDw17mAD6OOKSNu8' \
  --form 'files=@dialogue_long_clip.mp4' \
  --form 'assistant_id=4c798997-7be1-40ed-8de1-91a97ee08a66' \
  --form 'reference_audio=false' \
  --form 'files=@urls.md' \
  --form 'url=https://www.youtube.com/watch?v=_ySbzVXiwzQ' \
  --form 'url=https://www.youtube.com/watch?v=tlOyZSAZh2k'


