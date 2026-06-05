# Test of creating an avatar from a playlist with no specific target: 
# TEST Avatar ID: a6cdfa75-928d-4c23-a9f2-3495663db544

https://www.youtube.com/watch?v=gSNFJbgoaHI&list=PLQ-uHSnFig5M9fW16o2l35jrfdsxGknNB

curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-QiYFReD7tLoK8fri6n9ZmYcSpqmJDw17mAD6OOKSNu8' \
  --form 'assistant_id=a6cdfa75-928d-4c23-a9f2-3495663db544' \
  --form 'reference_audio=false' \
  --form 'reference_image=false' \
  --form 'all_speakers_target=true' \2
  --form 'url=https://www.youtube.com/watch?v=gSNFJbgoaHI&list=PLQ-uHSnFig5M9fW16o2l35jrfdsxGknNB'