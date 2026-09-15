# Example Creation of Paul Graham Avatar:

The user types in a name or presents an image clicks create.
deep research performs a search, finds a biography:
https://paulgraham.com/bio.html
uses this as a reference image and/or description

finds his "favorite or apt location"
https://www.google.com/search?client=ubuntu-sn&channel=fs&q=ycombinator+coordinates+real+world

fails to find coordinates and continues to search 
coordinates are found:
https://www.google.com/search?q=ycombinator+coordinates+real+world+location&client=ubuntu-sn&hs=N0W&sca_esv=8c75cec338c0fcad&channel=fs&sxsrf=APpeQnsnnVw1phxcgDhx2GPRGCrWAT8NDw%3A1789394363367&ei=u_2napmBFti3wN4PlNDyoAQ&biw=3786&bih=2027&ved=2ahUKEwjZypC7ne6WAxXYG9AFHRSoHEQQ4dUDegQIBhAM&uact=5&oq=ycombinator+coordinates+real+world+location&gs_lp=Egxnd3Mtd2l6LXNlcnAiK3ljb21iaW5hdG9yIGNvb3JkaW5hdGVzIHJlYWwgd29ybGQgbG9jYXRpb24yBRAhGKABMgUQIRigAUiQLlDwI1i4LHABeAGQAQCYAV2gAdEFqgEBObgBA8gBAPgBAZgCCqAC-gXCAgoQABhHGNYEGLADwgIFECEYqwKYAwCIBgGQBgKSBwM5LjGgB8QesgcDOC4xuAf2BcIHBTAuOC4yyAcRgAgB&sclient=gws-wiz-serp

Coordinates: 37.7601° N, 122.3882° W (Approximate)
Physical Address: 560 20th St, San Francisco, CA 94107 [1] (https://www.google.com/searchviewer/10?svid=CAwSHRIbCgNwdnESFENnMHZaeTh4TVd4a2EyNHhlR1J4GAo)

location of avatar is created

essays are identified:
https://paulgraham.com/articles.html
each link is recursively crawled
direct quotes are pulled
direct quotes are analyzed
a corpus of direct quotes for adapter training is created
facts and analysis results are stored in the vector store for retrieval
on hitting a threshold, the adapter is trained using the curated direct-quote dataset
when the training is complete, the adapter is attached and the model used for inference is altered to that endpoint (given enterprise plan to enable training and adapter inference use)

video reference is search for the avatar:
https://www.youtube.com/results?search_query=paul+graham

an interview with the avatar is identified (determination that the avatar speaks most often is made such that the video will be used to create reference audio for future diarization and instant voice cloning (professional voice cloning clips are curated if this is the personal avatar)):
https://youtu.be/5bxp78i96S8?si=k0V3qRVDHPxUbfdH

the content from that video is used to create a
reference audio clip

facts about the avatar are extracted
direct quotes are extracted
analysis is performed on the transcript to identify latent features of the avatar (emotional triggers, latent psychological traits, wants, needs, desires, fears, etc, psychological profiles, etc.)

the reference audio is now used in OTHER audio/video that is found and the new audio/video is diarized against this

if generative images/videos are enabled per tier, the content is created 

facts that are ambiguous found from deep research create cards that are used for manual curation from the owner / crowd
(crowd-sourced fact verification if this is a public avatar that is no long with us; primary sources trump popluar opinions of the crowd)

there is automated testing using the interview content to identify the response drift of the avatar from the real-world responses (this would be manual playtesting, how often do the responses sound like the real responses? do the responses feel like the real individual given the video content and interview transcript? Use LLM as a judge and automated offline and online testing)

IF THIS IS THE PERSONAL AVATAR:
search for all accounts that may be owned by the avatar. attempt to connect;

create an initial pull and distillation of the medium into text and subscribe;

on new posts, this information is distilled and used to update the avatar's identity (the facts are pulled, analyzed, used for the curated dataset, stored, and used for retrieval when appropriate during inference)

-----
The avatar is ready for inference and use to manage your digital life and communicate on your behalf.