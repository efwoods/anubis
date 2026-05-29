# Bugs
avatar does not identify self in images
images shared do not persist (I select another conversation and the image is not present; only the name of the image is present)
api key is shared when sharing avatars (should not be leaked)
conversation naming does not persist


# frontend conversation loading
@frontend 
@src/api/webapp.py 

I am sending: http://localhost:8501/?assistant_id=1bf5c443-a070-4a91-9872-74027eda0069&api_key=sk-RPE9F64MDhMK0vVVpkGyNenqKs5z73AVFngq8bqiKx8

Why am I seeing the following error:
New conversation

New conversation — not yet saved
🧑

Hey! Please tell me about yourself and what you can do for me.
assistant avatar
❌

HTTP 400: Error retrieving assistant information.
25MB per file

----------------------
Loading the url does not load the conversation history:

🤖 Studio Chat

Copies a link to this assistant and open chat (includes API key if set).
👋 Welcome to Studio Chat

Configure your settings with ⚙️ in the sidebar, then click ➕ New conversation to get started.

The assistant is determined by the URL: ?assistant_id=<your-id>

This bug occured then did not occur.

On load, the initial conversation is not started (there is no conversation open and a new conversation must be created). The share conversation button is sharing the api key. The api key must not be shared, seen, or alterable if the share url has been used to share conversations. 


# Copy Edit Paste Feedback buttons