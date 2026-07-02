NATURAL LANGUAGE EDITING GUIDANCE
---
Natural language:
CREATE FACT: update_self_identity_mem_from_user_txt
EDIT FACT: correct_identity_fact
DELETE FACT: correct_identity_fact


I need to be able to Use Natural Language to edit the documents in the avatar's identity
I need to retain the current integration with the frontend. 
correct_identity_fact needs to be separated into two different tools: edit fact and delete fact

@natural_language_edit_identity.json
@src/anubis/utils/tools/identity/identity_tools.py/update_self_identity_mem_from_user_txt 453: 723
@src/anubis/utils/tools/identity/identity_tools.py/correct_identity_fact 1784: 2066

update_self_identity_mem_from_user_txt is used to CREATE new facts about the avatar's identity as heresay from the user-creator only

editing facts is used when a user asserts a fact is incorrect and there are documents that contain the fact and other facts. That fact is then edited with the suggestion, the suggestion is editable by the user-creator, and is either edited or accepted (this is the standard that is currently practiced)

If the fact is asserted to be changed by the user (this event happened rather than this event never happened), and a document only contains that fact, then the fact is edited for the entire document. 

delete fact is used when a fact is asserted to be deleted by the user (the user says this never happened) and the document retrieved contains only the fact. if the document contains the fact and other facts, then the fact is edited rather than deleted. 

when the user says this never happened and no document exists, a new fact is created (this is a new fact the user is presenting to the avatar not to be confused with deleting the facts or editing the facts that currently exist)

EXAMPLES:

CREATE FACT:
you're 5' 6"
This fact is not in the vectorstore, the context of the conversation, or the retrieved documents in the current state of the graph. This is a new fact that is created and learned. It is important that the DOCUMENT is found (the fact within the context of the conversation does not indicate that the document exists within the state of the documents in the graph).

EDIT FACT:
you're 6' 1"
After creation, the document (the fact: "you're 5' 6") is edited to contain the information `I am 6' 1"` rather than `I am 5' 6"` This document may be within the current state or the vectorstore. 

DELETE FACT:
`Please forget your height` or `delete the information about your height`
the document containing `I am 5' 6"` or `I am 6' 1"` is suggested to be edited as `I don't have any information about my height` and the user may ELECT to delete the fact entirely. This is up to the discretion of the user when there is ambiguity. The avatar will respond saying, "I don't have any information about my height." when queried with, "How tall are you?"

EDIT FACT:
fact: `I am average for my height. I am 5' 6" `
becomes as a suggestion: `I am average for my height. I am 6' 1"`

AMBIGUOUS EXAMPLE (LEAN TOWARDS SUGGESTED EDITS OF FACTS RATHER THAN DELETION OF FACTS AND ALLOW THE USER TO DETERMINE IF A FACT SHOULD BE EDITED OR DELETED IN THE CASE OF AMBIGUITY):

USER SAYS: `You never wore braces`

CASE: THE FACT DOES NOT EXIST AS A DOCUMENT:
CREATE FACT: `You never wore braces`

CASE: THE FACT DOES EXISTS AS A DOCUMENT OF THE STANDALONE FACT:
EXISTING DOCUMENT FACT: `I wore braces once`
SUGGESTED EDIT THE FACT: `I never wore braces`

CASE: THE FACT DOES EXISTS AS A DOCUMENT OF THE STANDALONE FACT:
EXISTING DOCUMENT FACT WITH OTHER FACTS:
EXISTING DOCUMENT FACT: `I wore braces once. I remember wearing my braces and putting on my glasses.`
SUGGESTED DOCUMENT FACT EDIT: `I remember putting on my glasses.`
IN THIS CASE THE USER MAY SELECT TO DELETE THE FACT ENTIRELY HOWEVER THAT IS TO THE DISCRETION OF THE USER.


-----

# DOCUMENT STATEFULLNESS:
THE DOCUMENTS SHOULD BE STATEFUL; WHEN THE DOCUMENTS ARE RETRIEVED, THE DOCUMENTS THAT HAVE BEEN RETRIEVED AS WELL AS THE DOCUMENTS THAT ARE CURRENTLY IN LOADED IN THE GRAPH STATE NEED TO BE SCORED FOR SALIENCE TO THE CURRENT QUERY. DUPLICATE DOCUMENTS NEED TO BE REMOVED, AND DOCUMENTS THAT DO NOT MEET A THRESHOLD ARE ELIMINATED FROM THE CURRENT STATE OF DOCUMENTS IN THE GRAPH. 

RATHER THAN REMOVING ALL DOCUMENTS FROM THE STATE OF THE GRAPH AND SEARCHING FOR DOCUMENTS EACH TIME, THE DOCUMENTS NEED TO BE PERSISTED IN THE STATE (NOT DELETED OR REMOVED DURING LOAD CONSCIOUSNESS), COMPARED AGAINST THE RETRIEVED DOCUMENTS WITH DUPLICATES ELIMINATED, AND THRESHOLDED AGAINST THE USER QUERY FOR SALIENCE TO THE CONVERSATION. THIS ALLOWS TOOLS SUCH AS CREATE_FACT, EDIT_FACT, AND DELETE_FACT HAVE THE CURRENT DOCUMENTS AVAILABLE IN THE STATE OF THE GRAPH AND IF THOSE TOOLS ARE CALLED, THEY NEED TO BE CALLED FIRST, THE SALIENT DOCUMENTS EXTRACTED PER TOOL, AND THEN THE CONSCIOUSNESS IS LOADED. 