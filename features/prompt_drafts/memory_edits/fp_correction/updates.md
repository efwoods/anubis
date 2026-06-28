# Solution: select and edit the actual text in the documents; select multiple documents on which to apply a bulk edit

# addressing problems:
- [x] 1. tool is not called for each distinct fact that needs to be changed.
      → `FactCorrection` schema + tool docstring now instruct "call ONCE PER DISTINCT
        inaccurate fact" (mirrors `update_self_identity_mem_from_user_txt`).
- [x] 2. documents that are changed are not visible, unable to alter which documents are edited, unable to apply edits to individual documents, unable to select one or more documents with a bulk edit
      → interrupt payload now emits one editable item per match (`index`, `current_text`,
        `current_context`, `document_excerpt`, `suggested_text`, `suggested_context`,
        `recommended_action`, `default_action`). Each item carries an explicit per-document
        ACTION — `skip` (leave unchanged) / `accept` (suggested edit) / `edit` (owner's text) /
        `remove` (delete/redact) — defaulting to `skip`. Frontend panel renders each as its own
        block with a 4-way radio defaulting to "Leave unchanged" + editable text/context; a
        single "Apply my choices" applies each item's action, "Cancel correction" abandons all.
        Resume contract: `{type: apply|cancel, items:[{index, action, corrected_text?,
        correction_context?}]}`. Fixed the `current_fact`/`page_content` key mismatch that
        rendered every row as "(unnamed)".
- [x] 3. documents without the incorrect fact to be replaced are completely overwritten
      → per-document LLM suggestion sets `asserts_inaccurate_fact=False` for off-topic
        matches → `recommended_action="skip"`; and because every item DEFAULTS to `skip`,
        nothing is rewritten unless the owner explicitly opts in per document (so loose
        false-positive matches are safe even if the owner just hits Apply).
- [x] 4. documents with the incorrect fact and other facts are completely overwritten (should only select the incorrect fact within the document)
      → suggestion returns a MINIMAL in-place edit (only the offending span changes; the rest
        preserved verbatim) instead of a global overwrite.
- [x] 5. facts are not edited in place correctly in larger quote documents
      → each matched sentence gets its OWN replacement (empty ⇒ removed), so a quote keeps
        every other sentence instead of being flattened to the corrected fact repeated; the
        same substitution is mirrored into `scene_summary`/`user_context`.
- [x] 6. Documents with the incorrect incorrect fact are overwritten with the wrong incorrect fact.'
      → no single global correction is applied across heterogeneous matches anymore; every
        document carries its own resolved `corrected_text`, so the name fix can't land on a
        university doc.
- [x] 7. fact context is not preserved.
      → suggestion edits the document's ORIGINAL `fact_context` minimally (keeps still-true
        parts) rather than blanket-overwriting it with the new fact.

# addressing problems (learning):
- [x] 8. `update_self_identity_mem_from_user_txt` learns facts the avatar surfaced from its OWN
      retrieved consciousness (identity/quote transcripts in the system prompt) when the user
      merely ASKS about those topics — e.g. "tell me about U of T … spell your name" caused it to
      store "I grew up in Markham", "My name is Siobhan", etc. The fact never came from the user.
      → new runtime SAFEGUARD `_user_message_grounds_fact`: before storing, verify (model-backed,
        structured output) that the fact was derived from the user's MOST RECENT message only —
        not retrieved context, the assistant's own words, or an earlier turn. A question/request
        is never an assertion. Fails OPEN on empty message / model error (mirrors
        `_suggest_correction`) so genuine facts are never silently dropped.
      → NOTE (still open): the pre-existing dedup in this tool is non-functional for these cases —
        it only checks the `identity_memory` namespace + `assistant_identity_documents` state,
        while the offending facts live in the `quote`/`identity` namespaces; follow-up.

# Implementation (f-edit-memory):
- src/anubis/utils/tools/identity/identity_tools.py — message-grounding safeguard
  (`_user_message_grounds_fact`, `_latest_user_message_text`, `_extract_message_text`,
  `_UserMessageGroundsFact`, `_FACT_GROUNDING_SYSTEM_PROMPT`) gating
  `update_self_identity_mem_from_user_txt`; `ProposedFactEdit`, `_suggest_correction`
  (per-doc, concurrency-bounded, model-backed w/ graceful fallback), `ResolvedCorrection`,
  `apply_resolved_corrections`, per-sentence redaction, rewritten HITL decision handling.
- src/api/webapp.py — `/message/{id}/resume` accepts `match_edits` JSON → `matches` in the
  resume payload.
- frontend/studio_chat_app.py — per-document approve/reject panel; `api_resume_message`
  forwards `match_edits`.
- tests/unit_tests/test_correct_identity_fact.py — `_suggest_correction` stub fixture + new
  per-document edit / include-toggle / payload-shape tests (14 passing).
- tests/unit_tests/test_self_identity_fact_grounding.py — stubbed-model tests for the grounding
  safeguard: latest-message extraction, content-block flattening, verdict passthrough,
  question-only rejection, fail-open on empty message / model error (6 passing).


# Examples:

## 0. proper correction example: 
        {
          "metadata": {
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "corrected_from": "My name is Siobhan.",
            "document_id": "4aa1c6fe-ae8c-41be-bd53-ddf91c822b8d",
            "fact": "My name is Shivon.",
            "fact_context": "I was named Shivon Zilis by my parents at birth.",
            "user_id": "69e5e49980b783d7dff3012b"
          },
          "page_content": "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>I was named Shivon Zilis by my parents at birth.</FACT_CONTEXT><FACT>My name is Shivon.</FACT></FACT_CONTEXT_AND_FACT>",
          "type": "Document"
        },

## 1 tool is not called for each distinct fact that needs to be changed: 

This should have triggered the tool call for each distinct fact:


corrected_information: My name is Shivon.
correction_context: The user said my name is Shivon and that I never associated with the University of Alberta.
correction_kind: update
inaccurate_information: My name is Siobhan.

There should be two distinct fact corrections called from the context; correction_context: The user said my name is Shivon and that I never associated with the University of Alberta. 
FACT 1: The user said my name is Shivon
FACT 2: I never associated with the University of Alberta.

Only the documents with the incorrect fact should only have the fact corrected. The remainder of the fact should stay untouched within the document or quote. 

The following is the order of how the tool was called:

1
LangGraph
4:28 PM
14.14s
2
LangGraph
8:43 PM
22.37s
3
LangGraph
8:52 PM
22.55s
chat
0.00s
0.00s
resolve_human_message_images
0.00s
0.00s
anubis
22.54s
22.54s
LangGraph
22.54s
22.54s
load_consciousness
0.54s
0.54s
ChatPromptTemplate
0.00s
0.00s
think
21.98s
21.98s
LangGraph
21.94s
21.94s
PatchToolCallsMiddleware.before_agent
0.00s
0.00s
ConsciousnessRefreshGate.before_model
0.00s
0.00s
model
5.30s
5.30s
TodoListMiddleware.awrap_model_call
5.29s
5.29s
FilesystemMiddleware.awrap_model_call
5.29s
5.29s
SubAgentMiddleware.awrap_model_call
5.28s
5.28s
SummarizationMiddleware.awrap_model_call
5.28s
5.28s
DynamicConsciousnessPrompt.awrap_model_call
5.16s
5.16s
AnthropicPromptCachingMiddleware.awrap_model_call
5.16s
5.16s
ChatOpenAI
gpt-5.4-nano
5.11s
5.11s
TodoListMiddleware.after_model
0.00s
0.00s
tools
16.62s
16.62s
FilesystemMiddleware.awrap_tool_call
16.61s
16.61s
correct_identity_fact
16.61s
16.61s
4
LangGraph
8:56 PM
21.29s
anubis
21.29s
21.29s
LangGraph
21.28s
21.28s
think
21.27s
21.27s
LangGraph
16.14s
16.14s
tools
10.79s
10.79s
FilesystemMiddleware.awrap_tool_call
10.78s
10.78s
correct_identity_fact
10.78s
10.78s
ConsciousnessRefreshGate.before_model
0.00s
0.00s
tools
0.10s
0.10s
FilesystemMiddleware.awrap_tool_call
0.10s
0.10s
load_consciousness
0.10s
0.10s
ChatPromptTemplate
0.00s
0.00s
ConsciousnessRefreshGate.before_model
0.00s
0.00s
model
5.21s
5.21s
TodoListMiddleware.awrap_model_call
5.21s
5.21s
FilesystemMiddleware.awrap_model_call
5.20s
5.20s
SubAgentMiddleware.awrap_model_call
5.20s
5.20s
SummarizationMiddleware.awrap_model_call
5.19s
5.19s
DynamicConsciousnessPrompt.awrap_model_call
5.07s
5.07s
AnthropicPromptCachingMiddleware.awrap_model_call
5.07s
5.07s
ChatOpenAI
gpt-5.4-nano
5.02s
5.02s
TodoListMiddleware.after_model
0.00s

## 2. unable to view the documents that are being edited (unable to view the content that is being edited/select individual and/or bulk documents):
Your name is Shivon. You never associated with the University of Alberta.

🤖
✏️ I found existing fact(s) to correct — please confirm:

You flagged as inaccurate: My name is Siobhan.

Matched fact(s) that will be replaced:

(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
(unnamed)
Corrected fact

My name is Shivon.
Context

The user said my name is Shivon and that I never associated with the University of Alberta.



No file chosen

25MB per file
Type your message…

## 3. Documents without the incorrect fact are completely overwritten:

        {
          "metadata": {
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "corrected_from": "I grew up in Markham, Ontario.",
            "document_id": "84445b82-40bd-44a3-b876-7e0706a82a10",
            "fact": "My name is Shivon.",
            "fact_context": "I was named Shivon Zilis by my parents at birth.",
            "user_id": "69e5e49980b783d7dff3012b"
          },
          "page_content": "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>I was named Shivon Zilis by my parents at birth.</FACT_CONTEXT><FACT>My name is Shivon.</FACT></FACT_CONTEXT_AND_FACT>",
          "type": "Document"
        },

        {
          "metadata": {
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "corrected_from": "I have been in the Bay Area for a while.",
            "document_id": "ab905462-6453-4deb-96c9-fc2ddad4f554",
            "fact": "My name is Shivon.",
            "fact_context": "I was named Shivon Zilis by my parents at birth.",
            "user_id": "69e5e49980b783d7dff3012b"
          },
          "page_content": "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>I was named Shivon Zilis by my parents at birth.</FACT_CONTEXT><FACT>My name is Shivon.</FACT></FACT_CONTEXT_AND_FACT>",
          "type": "Document"
        },


## 4. documents with the incorrect fact and other facts are completely overwritten (should only select the incorrect fact within the document)
 {
          "metadata": {
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "corrected_from": "I get to work with the incredible people at University of Toronto and University of Alberta.",
            "document_id": "143cf8b2-db3b-4349-a945-40aee4cb99c3",
            "fact": "My name is Shivon.",
            "fact_context": "I was named Shivon Zilis by my parents at birth.",
            "user_id": "69e5e49980b783d7dff3012b"
          },
          "page_content": "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>I was named Shivon Zilis by my parents at birth.</FACT_CONTEXT><FACT>My name is Shivon.</FACT></FACT_CONTEXT_AND_FACT>",
          "type": "Document"
        },

        This should correct to remove the University of Alberta given the correct identification of the incorrect fact: 

        "that I never associated with the University of Alberta."

        Desired correction:

{
          "metadata": {
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "corrected_from": "I get to work with the incredible people at University of Toronto and University of Alberta.",
            "document_id": "143cf8b2-db3b-4349-a945-40aee4cb99c3",
            "fact": "I get to work with the incredible people at University of Toronto.",
            "fact_context": "ORIGINAL FACT CONTEXT SHOULD HAVE BEEN PRESERVED OR EDITED IN KIND",
            "user_id": "69e5e49980b783d7dff3012b"
          },
          "page_content": "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>I was named Shivon Zilis by my parents at birth.</FACT_CONTEXT><FACT>My name is Shivon.</FACT></FACT_CONTEXT_AND_FACT>",
          "type": "Document"
        },

## 5. Quote documents with the incorrect fact are not properly written with the correct fact in the scence summary, the page_content, and the user_context:

    "scene_summary": "The speaker (Siobhan) gives an overview talk describing Bloomberg Beta’s empirical JAST version, recounting her background and a personal anecdote, then summarising their machine intelligence investment focus and the history of their Machine Intelligence Reports (2014, 2015, and a preview for 2016). She outlines a framework for thinking about building enterprise machine intelligence—covering how to bring intelligence to the workforce and functions, referencing categories and examples—then hands off to her colleague James, who discusses, at a high level, the abstractions behind data science and machine intelligence by framing models as generated code, explaining differences from traditional software, and raising questions about managing, trusting, and deploying models in organizations.",


    "user_context": "What are the key takeaways from Siobhan’s empirical version of the JAST talk across the 2014, 2015, and upcoming 2016 Machine Intelligence landscape, and what framework does she say helps corporate audiences think about machine intelligence?",

    <!-- Example page content excerpt -->
    "page_content": "My name is Shivon. My name is Shivon. My name is Shivon. My name is Shivon. But But it's honestly, I would be homesick if not for machine intelligence. And the reason is because there's just such a wealth of talent. Uh and I get to work with the incredible people at University of Toronto and University of Alberta that brings me back home about ten times a year. My name is Shivon. Um One other fun fact, Minister Bains actually taught me a very interesting lesson by accident about Canadian government. My name is Shivon. You could see him from a mile away, and what that ended up teaching me was Canadian government is willing to take calculated risks that pay off because he looked fantastic. And so just an intro to what we do, we're Bloomberg Beta, we're an early stage venture capital fund that's solely focused on investing in founders that are transforming the future of work for the good. And within that we became obsessed with this area of machine intelligence and why that happened was sort of you know personal experience of me and the other folks at the firm. where, you know, we're knowledge workers just like probably most of the people in this room, and there was a subset of technologies emerging and the ones that had this sort of magical feel, they were making workflows more seamless, they were giving giving us new types of intelligence like what really is it about this subset of technologies? And this is dating back about four years ago, now we realised, you know, it's really it's really the technologies that are using learning algorithms that are the one that had this special panache. And so that led to You know, we're kind of a nerdy analytical fund doing this just deep dive into the world of machine intelligence to see just how much, you know, it was affecting the rest of the world. And so what I want to do is three things very quickly. My name is Shivon. So we've published the Machine Intelligence Report in 2014, we did one in 2015, and I'm gonna give you a sneak preview of what's coming up in 2016. We wanna give you a bit of a framework for how to think about this day. We wanna make sure that all of the corporate folks in the audience that are thinking about machine intelligence for the first time have a really good framework for how to do that. My name is Shivon. So 2014, how many of you have seen some sort of version of this chart? My name is Shivon. My name is Shivon. My name is Shivon. Um So the thing that surprised us in 2014, you know, really went into this heads down wondering are people using this technology and if so, what are they using it for? And the overarching takeaway from this slide was, oh    

    <!-- Full quote document -->

     {
          "metadata": {
            "adapter_acceptable": true,
            "adapter_formatted": false,
            "adapter_prompt": null,
            "analysis_acceptable": true,
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "audio_filename": "https://www.youtube.com/watch?v=-tQwzhHjAVI",
            "chunk_index": 0,
            "classified_situation": "tweets_or_quotes",
            "corrected_from": "My name is Siobhan and I will be giving the empirical version of the JAST talk, uh with my partner James Cham, um which is which is coincidental.; Where's the clicker?; Thank you.; So for those of you that don't know me, I grew up in Markham, Ontario and I've been in the Bay Area for a while now.; So thank you AI for my general life happiness.; The first time I met him, he was wearing these bright, bright, bright orange sneakers, and they were super, super cool.; I want to walk you through the history.; let my colleague James Cham accept up to do that.; Oh, wow.; First time I've ever felt like a celebrity.; Let's get it.; So in between all these angry founder emails I get this email that says hey Hey, Siobhan, I know we've only met briefly; you're working on this machine intelligence thing, can you come teach my MBAs about it?; And so here's what we've got going here.; Um it's it's a rough draft.",
            "created_at": "2026-06-25T23:54:39.329283+00:00",
            "document_id": "f337df4b-8b12-47eb-9a99-1776b4b49cdd",
            "end": 361.616,
            "filename": "https://www.youtube.com/watch?v=-tQwzhHjAVI",
            "filename_uuid5": "1b6d029e-bc0c-55d2-9d93-62cdc40534b2",
            "is_target": true,
            "item_job_id": "ae3a179c-a46d-4cdc-a1ca-67c0166fcde5",
            "master_job_id": "22c777d8-dadc-40d1-92a9-9e539a0cfb12",
            "namespace": "quote",
            "namespace_filename": "1b6d029e-bc0c-55d2-9d93-62cdc40534b2",
            "processing_task_id": "8758844e-7b52-4332-bfb1-d091e9a9c593",
            "redacted_sentences": [
              "My name is Siobhan and I will be giving the empirical version of the JAST talk, uh with my partner James Cham, um which is which is coincidental.",
              "Where's the clicker?",
              "Thank you.",
              "So for those of you that don't know me, I grew up in Markham, Ontario and I've been in the Bay Area for a while now.",
              "So thank you AI for my general life happiness.",
              "The first time I met him, he was wearing these bright, bright, bright orange sneakers, and they were super, super cool.",
              "I want to walk you through the history.",
              "let my colleague James Cham accept up to do that.",
              "Oh, wow.",
              "First time I've ever felt like a celebrity.",
              "Let's get it.",
              "So in between all these angry founder emails I get this email that says hey Hey, Siobhan, I know we've only met briefly; you're working on this machine intelligence thing, can you come teach my MBAs about it?",
              "And so here's what we've got going here.",
              "Um it's it's a rough draft."
            ],
            "scene_summary": "The speaker (Siobhan) gives an overview talk describing Bloomberg Beta’s empirical JAST version, recounting her background and a personal anecdote, then summarising their machine intelligence investment focus and the history of their Machine Intelligence Reports (2014, 2015, and a preview for 2016). She outlines a framework for thinking about building enterprise machine intelligence—covering how to bring intelligence to the workforce and functions, referencing categories and examples—then hands off to her colleague James, who discusses, at a high level, the abstractions behind data science and machine intelligence by framing models as generated code, explaining differences from traditional software, and raising questions about managing, trusting, and deploying models in organizations.",
            "source": "https://www.youtube.com/watch?v=-tQwzhHjAVI",
            "speaker": "avatar",
            "start": 4.756,
            "synthetic": false,
            "target_name": "avatar",
            "total_chunks": 2,
            "type": "text",
            "user_context": "What are the key takeaways from Siobhan’s empirical version of the JAST talk across the 2014, 2015, and upcoming 2016 Machine Intelligence landscape, and what framework does she say helps corporate audiences think about machine intelligence?",
            "user_context_synthetic": true,
            "user_id": "69e5e49980b783d7dff3012b",
            "vectorstore_acceptable": true
          },
          "page_content": "My name is Shivon. My name is Shivon. My name is Shivon. My name is Shivon. But But it's honestly, I would be homesick if not for machine intelligence. And the reason is because there's just such a wealth of talent. Uh and I get to work with the incredible people at University of Toronto and University of Alberta that brings me back home about ten times a year. My name is Shivon. Um One other fun fact, Minister Bains actually taught me a very interesting lesson by accident about Canadian government. My name is Shivon. You could see him from a mile away, and what that ended up teaching me was Canadian government is willing to take calculated risks that pay off because he looked fantastic. And so just an intro to what we do, we're Bloomberg Beta, we're an early stage venture capital fund that's solely focused on investing in founders that are transforming the future of work for the good. And within that we became obsessed with this area of machine intelligence and why that happened was sort of you know personal experience of me and the other folks at the firm. where, you know, we're knowledge workers just like probably most of the people in this room, and there was a subset of technologies emerging and the ones that had this sort of magical feel, they were making workflows more seamless, they were giving giving us new types of intelligence like what really is it about this subset of technologies? And this is dating back about four years ago, now we realised, you know, it's really it's really the technologies that are using learning algorithms that are the one that had this special panache. And so that led to You know, we're kind of a nerdy analytical fund doing this just deep dive into the world of machine intelligence to see just how much, you know, it was affecting the rest of the world. And so what I want to do is three things very quickly. My name is Shivon. So we've published the Machine Intelligence Report in 2014, we did one in 2015, and I'm gonna give you a sneak preview of what's coming up in 2016. We wanna give you a bit of a framework for how to think about this day. We wanna make sure that all of the corporate folks in the audience that are thinking about machine intelligence for the first time have a really good framework for how to do that. My name is Shivon. So 2014, how many of you have seen some sort of version of this chart? My name is Shivon. My name is Shivon. My name is Shivon. Um So the thing that surprised us in 2014, you know, really went into this heads down wondering are people using this technology and if so, what are they using it for? And the overarching takeaway from this slide was, oh my goodness, there's already a lot happening. And the other thing that surprised us was just the breadth of activity. You're seeing it affecting every vertical. And One of the things that happens when you publish a whole bunch of these reports is you end up getting a whole bunch of inbound from people that are interested. And and what happened in 2014 with the inbound, it was primarily angry founders that weren't on the landscape, so we fixed that. Uh but there was one really really really good thing that happened from it, and I wouldn't be here today if it if it weren't for the future looking efforts of a J Agarwal. My name is Shivon. And I was like, well, sure, any excuse to go to Toronto, but You know, are you sure you want to teach MBAs about this? And I'm sure they haven't even heard of machine learning yet. And he's like, well, you know, that's true, but I think that this trend is going to be bigger than anybody realised. You're going to have a whole bunch of technologists coming up and what's not going to be there is going to be the business leaders that can help accelerate these technologies. And he's absolutely right. So the machine learning cohort this year for the Creative Destruction Lab has 50 companies and one of the most magical things that I found about the programme that doesn't really exist anywhere else is you've got these incredible technical founders that are being paired with these business leaders that have a savviness in machine intelligence, which is kind of this beautiful marriage. So jumping forward to 2015, two big takeaways here. One is volume activity in the startup ecosystem exploded for sure. This is a much busier chart that's showing a much smaller subset of what's actually happening in machine intelligence. And the second bit here was it was the first time we saw the emergence of autonomous systems. And so, you know, self-driving cars, autopilot drones, things like industrial machinery starting to move in a more human and dynamic way. A and then the software portal area of that was we saw these agents. And so the way to think about the agents or if you wanna be mean you can call them chatbots, I think they're more than that, um is you have these pieces of software that can actually understand us in some ways, transact for us and act in inter-temporal ways. And so that that was the first time we saw the emergence there. And so dating up to 2016, you guys get the sneak peek we're not publishing until next week, but this is the home team advantage. The big, big, big difference, and Ajay, you alluded to this in your talk, was I do think this is the inflection point here. And the reason I think that, at least from sort of a broad-based machine intelligence perspective, is we get thousands and thousands of inbound messages as a result of these landscapes. And there's been a stark contrast in the type of person who's reaching out. So in the first landscape, it was mostly founders and academics. Then there were investors that started becoming interested and for the last year it's been primarily corporate executives trying to figure out exactly how they're going to bring machine intelligence capabilities to their own corporations. And so for this year's landscape it really comes with an eye for okay how do I actually do this. My name is Shivon. My name is Shivon. So for forgive any misalignments, but the way to think about this isn't is isn't three parts. And so you've got that top part there.",
          "type": "Document"
        }


        

## 6. Documents with the incorrect incorrect fact are overwritten with the wrong incorrect fact.

        {
          "metadata": {
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "corrected_from": "I get to work with the incredible people at University of Toronto and University of Alberta.",
            "document_id": "143cf8b2-db3b-4349-a945-40aee4cb99c3",
            "fact": "My name is Shivon.",
            "fact_context": "I was named Shivon Zilis by my parents at birth.",
            "user_id": "69e5e49980b783d7dff3012b"
          },
          "page_content": "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>I was named Shivon Zilis by my parents at birth.</FACT_CONTEXT><FACT>My name is Shivon.</FACT></FACT_CONTEXT_AND_FACT>",
          "type": "Document"
        },


## 7. fact context needs to be edited in alignment and the original fact context should be preserved when appropriate so as not to completely overwrite the fact context.
<!-- Expected Edit -->
        {
          "metadata": {
            "assistant_id": "60590261-21f9-4110-854c-f1819ea1c1fe",
            "corrected_from": "I get to work with the incredible people at University of Toronto",
            "document_id": "143cf8b2-db3b-4349-a945-40aee4cb99c3",
            "fact": "I get to work with the incredible people at University of Toronto.",
            "fact_context": "The user gave a presentation stating that they work with the people at the University of Toronto.", <-- The fact context needs to be edited to preserve facts that are true rather than blanketly overwritting the entire context -->
            "user_id": "69e5e49980b783d7dff3012b"
          },
          "page_content": "<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>I was named Shivon Zilis by my parents at birth.</FACT_CONTEXT><FACT>My name is Shivon.</FACT></FACT_CONTEXT_AND_FACT>",
          "type": "Document"
        },