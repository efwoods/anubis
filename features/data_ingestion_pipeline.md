# video/audio/url pipeline

https://claude.ai/chat/2d75c991-8396-4b6f-9695-3cc8491cdc5e

# upload video of life story; 
# add auto biography (book or other medium)

-------------
1. I need to finalize the document ingestion pipeline to accept all forms of text (markdown, text, json, csv, pdf) 
2. and conversion to text (audio to text, video to audio to text, images to textual description; 
3. uploading of reference audio file for speaker identification; (API ENDPOINT)
4. uploading of reference image for target identification in images). (API ENDPOINT)

There will be an upload url endpoint that will accept:
one or more of any media type:
the api endpoint will use this specifically for uploading of identity and the messaging endpoints will be able to use the api endpoint or logic for the same capability in message (sending a picture in chat saying this is a picture of you will allow the image to be described using a reference image if available to describe the image and upload the description to the vectorstore with the correct namespace for RAG. The api endpoint will accept an image and pipeline the image through this process without routing determination. The point is that the langgraph logic can use the API endpoint or the logic of the API endpoint)

# Data ingestion
I need to be able to accept unstructured text and classify a portion of the text with a model with structured output. 
I also need to be able to accept a csv or json and convert that into a string and process the same for classification to understand the situation.

URLS need to be processed into text and the text needs to be analyzed for situation. (using Langgraph Document Loaders)

----


# Classification Process:
@schema.py (185-453) @schema.py

# Refactoring Data Processing 
I'm currently processing the document type in the webapp, then again here: process_uploaded_files_and_label_media_type, then again in convert_media_list_to_text_document (where the document situation is analyzed then chunked and marked for the appropriate namespace), then I am indexing. The documents are embedded on page content. 
@webapp.py (1193-1293) 

I should be able to accept a document of any type, if text of any sort or url of any sort, then I need to have a large text sample that is classified for situation via a model with structured output. video is pipelined to separate processing (conversion to audio then conversion to text). audio is converted only to text. Images are described with a model with structured output as text and use a reference image if available to identify the target in the image and describe the target. All text is then sent to the same text processing. 

Initial preprocessing: Text pre-processing involves content situational analysis, target identification, and structured formatting.  


after pre-processing: 

Then there are parallel output steps from pre-processing: (analysis, vectorstore indexing, and adapter formatting). 
