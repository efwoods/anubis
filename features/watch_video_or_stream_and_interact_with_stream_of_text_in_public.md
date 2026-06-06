# Avatar watches a video: 
given a video, describe multiple frames of the video to build a conversation that is summarized iteratively after a point where the conversation is a description of frames of the video. The frames are timestamped and the audio is transcribed and included in the description. This is a video to image/audio textual description and eventual summarization. Feature includes target identification in the footage and description of the individual and analysis of the individual.

# Listen to music or a podcast:
the audio is converted into text and iteratively consumed as a conversation. 

# watch a live stream:
The stream is processed into text as above iteratively and the text is added as a textual message in the conversation

# Interact with a live stream
ignore notify responde to a live chat:
there are messages that are sent when the avatar is in a twitch or otherwise chat. 
The messages may be directed at the avatar (mentioned).


The avatar may ignore a message, respond, to a message, or notify the user to respond to the message (the response preference is stored in the vectorstore and returned)
The messages are a stream and each message is triaged (do I ignore this message? do I respond to the message? I need to respond to this message and I don't know what to say, notify the user to ask for an edit and respond)

# Live-stream text chat; triage a block of text
# Mentioned directly is triaged in response

