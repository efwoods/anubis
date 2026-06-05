# Shivon Zilis TEST: (9549b243-4774-45fa-8234-5137d6dd5bcc)
## Reference audio:
https://www.youtube.com/watch?v=-tQwzhHjAVI

playlist (single speaker, dialogue, multi-video playlist with target)
https://www.youtube.com/watch?v=CkUcCcRq_eM&list=PL9rU625vkl4UlyAT5THtDV3cOB2KOkASX


# reference documents:
/home/user/gh/anubis-project/wt/f-merge/_test_data/Machine Intelligence - Shivon Zilis.pdf
/home/user/gh/anubis-project/wt/f-merge/_test_data/confirmed_search_results_list.txt

# Test reference audio (there is a target speaker followed by a non-target speaker; target is monologue; non-target does not reference target)
curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-QiYFReD7tLoK8fri6n9ZmYcSpqmJDw17mAD6OOKSNu8' \
  --form 'files=[""]' \
  --form 'url=https://www.youtube.com/watch?v=-tQwzhHjAVI' \
  --form 'assistant_id=9549b243-4774-45fa-8234-5137d6dd5bcc' \
  --form 'reference_audio=true' \
  --form 'reference_image=false' \
  --form 'all_speakers_target=false'



# Test playlist (four videos; dialogue interview, biographical facts without target, single target monologue, short dialogue between target and non-target):
https://www.youtube.com/playlist?list=PL9rU625vkl4UlyAT5THtDV3cOB2KOkASX


curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-QiYFReD7tLoK8fri6n9ZmYcSpqmJDw17mAD6OOKSNu8' \
  --form 'files=[""]' \
  --form 'url=https://www.youtube.com/playlist?list=PL9rU625vkl4UlyAT5THtDV3cOB2KOkASX' \
  --form 'assistant_id=9549b243-4774-45fa-8234-5137d6dd5bcc' \
  --form 'reference_audio=false' \
  --form 'reference_image=false' \
  --form 'all_speakers_target=false'

# Test multi-media upload (pdf, url to grokipedia (biographical information))
curl /update_avatar_identity_with_media \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --header 'API-KEY: sk-QiYFReD7tLoK8fri6n9ZmYcSpqmJDw17mAD6OOKSNu8' \
  --form 'files=@Machine Intelligence - Shivon Zilis.pdf' \
  --form 'assistant_id=9549b243-4774-45fa-8234-5137d6dd5bcc' \
  --form 'reference_audio=false' \
  --form 'reference_image=false' \
  --form 'all_speakers_target=false' \
  --form 'files=@confirmed_search_results_list.txt'
