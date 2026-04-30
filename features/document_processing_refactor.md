I'm processing the document type in the webapp, then again here: process_uploaded_files_and_label_media_type, then again in convert_media_list_to_text_document (where the document situation is analyzed then chunked and marked for the appropriate namespace), then I am indexing the documents through a batching process into the vectorstore. 

I want to do this in a single step (extracting the document metadata for example) before 

The document needs to be in a format for analysis, 
vector store versions of the text need to be formatted such as to strictly store only information that are facts about the target or direct quotes from the target, and there needs to be an analysis step to identify if the content would be suitable for adapter dataset creation and creation of the prompt questions and pairing with the answers (if a monologue then a prompt question is created, if a series of distinct quotes then a series of paired provoking questions are generated, if facts about the target then this format is not acceptable for adapter dataset creation) and the dataset would need to be created and stored and training would need to be triggered after a sufficient amount of data is created. 

There is the metadata labeling of the original content, 
there is a conversion of any media to text (images to llm description, audio to text, video to audio to text), then situational analysis of the text content (how many speakers, direct quote from the target, information about the target, tweets only or monologue, dialogue or multi-speakers, biographical information, information about the target, direct quotes from the target, mixed content of all the previous). 
Then once the original metadata has been collected, 
media has been distilled to text, 
the situational content of the media is known, 
the content has been properly formatted appropriate to the situation there are three parallel processes (nodes) analysis, vectorstore indexing, and adapter dataset creation. These nodes have sub-steps (there are many different types of simultaneous analysis, adadpters create questions if the content is appropriate)
then the analysis is finally stored in the correct namespace in the vectorstor for retrieval (index_docs as a doc) and the vectorstore documents are indexed and the adapter dataset is uploaded and the quality analysis dataset is uploaded to langsmith for analysis (same as adapter training dataset). This is the full lifecycle of the uploading of media and distillation of personal media to an indexed vectorstore for retrieval. 

