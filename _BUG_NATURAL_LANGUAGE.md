NATURAL LANGUAGE EDITING:
CONTEXT:

user informed the avatar of their favorite color.
avatar learned this information.

user informed the avatar to forget this information.
avatar learned they do not have a favorite color as a fact.

user informed the avatar of a their favorite color (different color).
avatar called tool update_self_identity_mem_from_user_txt with the fact from the learned document "I don't have a favorite color".

The `fact_shared_about_the_assistant_from_the_user` is a fact from earlier in the context. The context clearly states:

    fact_context: The user said: "Your favorite color is blue."
    fact_shared_about_the_assistant_from_the_user: I don’t have a favorite color.


update_self_idenitity_mem_from_user_txt needs to correctly learn facts from the previous message only.

fact_shared_about_the_assistant_from_the_user is incorrect. 

If the fact already exists in the vectorstore and needs to be altered, edit_identity_fact needs to be called NOT update_self_identity_mem_from_user_txt.

if the fact does not exist in the vectorstore, only then does the update_self_identity_mem_from_user_txt need to be called with the correct fact that only comes from the most recent user message with context from the most recent user message for that fact.

if the fact alread exists in the database and that information to be learned is already in the database, then nothing needs to be edited or changed.

editing facts is used when a user asserts a fact is incorrect and there are documents that contain the fact and other facts. That fact is then edited with the suggestion, the suggestion is editable by the user-creator, and is either edited or accepted (this is the standard that is currently practiced)

If the fact is asserted to be changed by the user (this event happened rather than this event never happened), and a document only contains that fact, then the fact is edited for the entire document. 

The vectorstore needs to be searched for the fact when the intent to create a new fact, edit a fact, or delete a fact is the intended tool call. given the above scenarios and scenarios listed in _NATURAL_LANGUAGE_FACT_EDITING_GUIDANCE.md, the tools update_self_identity_mem_from_user_txt, edit_identity_fact, delete_identity_fact all need to be called when appropriate with the correct factual information given the presence or absence of the fact within the vectorstore.



