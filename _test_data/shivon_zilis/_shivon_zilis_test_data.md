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
  --form 'treat_every_speaker_as_target=false'



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
  --form 'treat_every_speaker_as_target=false'

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
  --form 'treat_every_speaker_as_target=false' \
  --form 'files=@confirmed_search_results_list.txt'


{
  "job_id": "6a0ce994-305b-4a03-a504-35cf17fdf655",
  "status": "queued",
  "progress_url": "/media_job/6a0ce994-305b-4a03-a504-35cf17fdf655/progress",
  "cancel_url": "/media_job/6a0ce994-305b-4a03-a504-35cf17fdf655/cancel",
  "items_accepted": 2,
  "filenames": [
    "Machine Intelligence - Shivon Zilis.pdf",
    "https://grokipedia.com/page/Shivon_Zilis"
  ],
  "items": [
    {
      "job_id": "b116d3ad-c467-4106-b94c-559fc57972be",
      "filename": "Machine Intelligence - Shivon Zilis.pdf",
      "status": "queued",
      "progress_url": "/media_job/b116d3ad-c467-4106-b94c-559fc57972be/progress",
      "cancel_url": "/media_job/b116d3ad-c467-4106-b94c-559fc57972be/cancel"
    },
    {
      "job_id": "8f2d56cb-5992-445e-a0c7-cb582bfa5421",
      "filename": "https://grokipedia.com/page/Shivon_Zilis",
      "status": "queued",
      "progress_url": "/media_job/8f2d56cb-5992-445e-a0c7-cb582bfa5421/progress",
      "cancel_url": "/media_job/8f2d56cb-5992-445e-a0c7-cb582bfa5421/cancel"
    }
  ],
  "playlists_expanding": 1,
  "message": "Media processing started; enumerating 1 playlist(s) in the background"
}