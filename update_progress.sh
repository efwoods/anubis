#!/bin/bash

if [ -z "$1" ]; then
  _MESSAGE="Please summarize the changes made since the last commit."
else
  _MESSAGE="$1"
fi

git diff main > progress.txt

curl http://localhost:8123/message/00da3ee4-e091-4f7e-b958-f2cc7f13e19f \
  --request POST \
  --header 'Accept: application/json' \
  --header 'Content-Type: multipart/form-data' \
  --form "message=${_MESSAGE}" \
  --form 'your_name=' \
  --form 'your_description=' \
  --form 'conversation_title=' \
  --form 'files=@progress.txt' \
  --form 'thread_id=' \
  --form 'stream=false' \
  --form 'feedback=false' \
  --form 'like=false' \
  --form 'dislike=false' \
  --form 'user_timezone=' > progress_summary.txt

git commit --allow-empty -F progress_summary.txt && git push
rm progress.txt progress_summary.txt