# when using update_identity_using_media
proprietary documents are stored under the document namespace and pulled under the  document namespce into the retrieved knowledge section of the system prompt

non proprietary documents are stored under the document namespace and pulled under the document namespace into retrieved knowledge section of the system prompt

analyzed information from documents (metadata about the user analyzed with separate system prompts from the documents) that are non-proprietary are stored under the identity namespace

Sequence:
update_identity_using_media -> stored under a namespace -> retrieved during load_consciousness (with information relevant to the query) -> injected into a system prompt -> used as a system message

