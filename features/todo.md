
# Metering 72 tokens

# create synthetic question dataset from questions with pipeline
# adapters 

# POINT OF IMPROVEMENT
- [ ] pre-processing: (reduce type 1 transcription errors)
- [ ] retrieval: increase quality of responses (score)
- [ ] quality: implement PCA and centroid for metrics that point to the quality of the response rather than the length
- [ ] adapter: (create, train, attach, infer on local test)
- [ ] analysis pipeline applied

# suggested features Z:
- [ ] CONTINUOUS LEARNING AND IMPROVEMENT (SYSTEM LEARNS FROM INTERACTIONS; POSITIVE SENTIMENT, ENGAGEMENT, RATINGS SYSTEM; LEARNS WHAT FEELS REAL TO USERS)
- [ ] CONVERSATION SUBLTIES (HUMOR, SARCASM, EMOTIONAL CUES)
- [ ] PERSONALIZATION (TAILOR RESPONSES TO USER PREFERENCES, COMMUNICATION STYLE, PREVIOUS INTERACTIONS)
- [ ] MODERATE CONVERSATIONS

'
# Obvious problems
 - [] Transcription is not always accurate: “empirical version of the JACE talk.”: https://www.youtube.com/watch?v=-tQwzhHjAVI
 - [] Response structure sounds like chatgpt: response; suggested list of advice; probe to continue the conversation
 - [] Testing is slow
 - [] Fine tuning on known responses to reduce false positives; 
 - [] Duplicates in system prompt of retrieved assistant_identity documents
  
The data is inaccurate; duplicated, responses are incorrect and inauthentic.


# Goal: Accurate relevant data and correct authentic responses.

- [ ] edit memories
- [ ] twitter login: https://x.com/shivon
- [ ] fragments in dataset from PDF: 11/5/25, 11:46 PM; Page 14 of 10; Machine Intelligence - Shivon Zilis.pdf
- [ ] transcription fragments from youtube: "so you're"; https://www.youtube.com/watch?v=WV329HQvzUw
- [ ] 


# COMPARING AUTHENTICITY OF GENERATED RESPONSES
data/elon-musk-dataset/elon_musk_tweets.csv

URL processing is unclean
pdf processing is unclean
processing media is incorrect/needs edits

cannot process:
wikipedia
twitter

instagram
spotify (music)

biographical and quotes: https://fallout.fandom.com/wiki/Curie

automate preprocessing collecting and preprocessing of llm data for responses (chatgpt, claude, grok) best way to extract adapter information 

https://www.google.com/search?client=firefox-b-1-m&q=mahalanobis+plot+i+need+to+know+to+what+degree+the+tweets+i+make+are+authentic&zx=1774299854965&no_sw_cr=1&fbs=ADc_l-bZnt6jMmErT-KRarIgXyuyzQTJF2PfkaOcTEalUPOVQOtlzownaoM46W2pScWjtBeiLTI6jl7uuBUhx4iC4iPTTJimJc6C_VwSLMaMy1J-5nzkD6p6csmb37ID92SDE-exCF46WPNzQdNHd1fpq-TmW4kQlGQ7a2_aVG30JAA7JPbzRX-wBWFkIAoqZbjb1_qIrzkSk-L-rW19pSfNoCRBAQlFCx-IDUv1N5OidEAKGpywaXo&aep=10&ntc=1&mstk=AUtExfCUxcYygjoG8VdAMpRFVH-POCqcz91VZBq-or3lPPszQWPRwtEh6ZqyxMKVi1g8HBMbN_C-9f9Vr-5_RAH3tN8L8TX9FtjC0WkcSGazwcp4o7yOg5rwZuRE8W6mRtDC9d_7Wh7lB3LFxApf_v54WpvAYQgVvOyE2co_AVkKrXYsBFO7mBnQWpqlzI5Bn1R-XGImR-zoSHCL_Oi3FJxxrOt57rSoobhErRlPVyZ4hSP0BSGM1VyYR07r0YduPyE8zSL8RBsslmOYnvKRcFxu3YTocgNOl-8xumE&csuir=1&aioh=3&mtid=86rBabO6O-DdptQP7uqL6A0&udm=50#lfId=ChxjMe

##############

Iterate through a playlist: https://www.youtube.com/watch?v=gIF_D6iUusU&list=PL9rU625vkl4XSOQDZxdFhVZgows3FJiKH&index=6


Tuning style (so the avatar does not feel like a Fact Machine; does not use Chatgpt phrasing)
Evaluating generated text for style and factual correctness and consistency within a thread
<!-- Emotions (basic emotions as a feature if requested) -->

zip file upload of media

linktree crawling of media

deep research of avatar using a name

send optional context with any uploaded media; the context is used to parse and classify the documents 

biographical and quotes: https://fallout.fandom.com/wiki/Curie

diarize songs to text

extract facts and quote from wikis: https://fallout.fandom.com/wiki/Curie

Clean extracted facts from websites or other textual media (currently more thought needs to be placed into the quality of the format of the body of the scraped media)

process upload text documents that are quotes/tweets: https://x.com/lexfridman

<!-- compare and contrast adapter use (deprecation is a key signal that this is most likely not the way) -->


## process upload text documents that are multi-speaker with the target: (target: Lex Fridman: )

https://www.youtube.com/watch?v=hLZ6PACCBy8

## process upload text documents that are multi-speaker that reference the target: (target: Grant Imahara: )

https://www.youtube.com/watch?v=xRHFylOgeJA

## process upload text documents that are multi-speaker that reference the target with the target direct quotes: (autobiographical about the target with direct quotes and multi-speaker)
    =https://www.youtube.com/watch?v=Px_5Z0pPlPc&t=2s


""" # Multi-file type bulk upload """
# process a zip file of information
# process a link tree: https://linktr.ee/ev0ra?utm_source=linktree_profile_share&ltsid=217800c7-7f92-45dd-88f6-9f364d64275c
                                                                                                
# deep research agent: (https://docs.langchain.com/oss/python/deepagents/deep-research#build-a-deep-research-agent); https://github.com/langchain-ai/open_deep_research; https://academy.langchain.com/courses/take/deep-research-with-langgraph/texts/67644896-getting-set-up; https://github.com/langchain-ai/deep_research_from_scratch (provide a name to research, validate the individual from the retrieved findings and process the media (distill to text then process the text as above))

""" Verification """

# logging into your own social media to verify that the social media account is yours and performing an initial pull of the media (processed as above) then subscribing to the social media channel for updates that are then pulled and processed

# perform all of the previous steps in a message chat (tool use for learning information about the avatar is appropriately triggered then the graph to process media is appropriately implemented)

# ontop of data processing there is:

# - psycho analyzing the data with respect to MEYERS briggs personality scores, OCEAN analysis, further inferred traits (emotional triggers, relationships, factual qualities of the individual, behaviors, lexicon of vocabulary, rate of speaking, style of speaking)
# - fine-tuning adapter with the creation of a dialogue when available to create prompt questions or use a dialogue to create a fine-tuned adapter to capture the speaking style and behavioral choices of the avatar

# This realizes Success: speak like the person in real time with audio of the individual; write like the person; infer the thoughts (that's what I thought!)


----

# Dataset Creation:

## Question and Answer pairs focused on target direct quotes
target speaker statements are collected and stored as quote documents
target speaker statements are used with non-target statements to create a question and answer dataset when those prompts genuinely exist from the content. when there is not a preceeding statement, a prompt based on the target statement is generated using a model with structured output. Then there is a question and answer pair used for adapter training and evaluation. 

## Multi-turn dialogue conversation dataset creation
Then the non-target questions are coalesced such that during the entire conversation, there are only two speakers regardless of the number of speakers that are present in the original media. The target is the assistant and any non target is the user such that there are only individual turns between the target and the non-target. for example:

- 1. Q and A with direct quotes
- 2. Full Q and A dialogue


## Transcription Dialogue:  
https://www.youtube.com/watch?v=CkUcCcRq_eM

## Text labeled multi-speaker
### Challenge: scan for a particular individual; create datasets focusing on that target
https://scrapsfromtheloft.com/movies/gray-man-2022-transcript/

## Reference Corpus:
### Challenge: scan reference corpus for direct quotes and immediately preceeding prompt questions. If there are no preceeding prompt questions, generate a synthetic prompt. 

<!-- /home/user/gh/anubis-project/anubis/data/bible/king_james/king_james_bible.txt -->

# Dataset Creation
        @src/aubgraphs/process_media_graph/utils/nodes.py function process_adapter_documents; This function should be using the genuine previous prompts already created to create a question and answer dataset. Only if there is not a prompt to the target statement does the prompt need to be synthetically generated. The questions need to be saved such that the
        questions may be presented to the synthetic llm and compared against the ground truth answers and the question and answer dataset used for training adapters is stored in the store under the q_and_a namespace; That is to say the langsmith for quotes dataset is a dataset with the genuine questions and the real responses when available and a synthetic prompt
        question when there is none available. This needs to be saved under the store namespace user_id, assistant_id, langsmith_factual_q_and_a, source_filename_mapped_to_uuid5; There needs to be a distinct question and answer dataset using the genuine prompts and the genuine responses when available otherwise a synthetic prompt needs to be presented if none such prompt
      is available. This question and answer dataset needs to be formatted as an llm_single_turn_dataset and saved as a json string under the store namespace (user_id, assistant_id, q_and_q_adapter, source_filename_uuid5); there should also be a multi-turn dataset with a single conversation formatted into llm_multiturn_dataset_one_conversation found in
        src/anubis/utils/dataset/formatting.py as well. This needs to be saved under (user_id, assistant_id, multi_turn_dataset_adapter, source_filename_uuid5). The langsmith dataset is used to test factual correctness, the question and answer dataset is used to create adapters and the multi-turn dataset will be used to train the adapter to attune behavior. These should
        be saved in the runtime store. these should not be saved to disk. The datasets need to be saved under jsonl format to train the adapters or to load to langsmith for online testing of factual correctness of the responses to the questions.
