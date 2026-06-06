"""Ad-hoc check: does the avatar extract EVERY distinct fact from a shared story
and call ``update_self_identity_mem_from_user_txt`` once per fact (instead of
summarizing into a single fact)?

Run from repo root:  python scripts/test_multifact_learning.py
Requires OPEN_AI provider keys in .env / .env.dev.
"""

import asyncio

from dotenv import load_dotenv

load_dotenv(".env.dev")
load_dotenv(".env")

from langchain_core.messages import HumanMessage

from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder
from src.anubis.utils.model import init_model
from src.anubis.utils.tools.identity.identity_tools import (
    learn_information_about_the_user,
    update_self_identity_mem_from_user_txt,
    wrap_fact_with_context,
)

STORY = (
    'I need you to learn that you are a Man. That the day you were going for '
    'glasses, you decided to pick up your glasses before you saw the movie, '
    '"Crouching Tiger, Hidden Dragon", that you thought is was good that you got '
    'your glasses before you saw the movie, because everything was now clear! You '
    'could read the signs in the back of the Walmart (you told this to your mom). '
    'You could see the individual leaves on the trees! She felt so bad that you '
    "couldn't see at the time, but you told her it's okay. You loved watching "
    'action movies and science fiction movies with your mom. You were her "guy". '
    'You were her "buddy".'
)


async def main() -> None:
    model = init_model(
        tools=[update_self_identity_mem_from_user_txt, learn_information_about_the_user],
        tool_choice="auto",
    )

    # Render the real production system prompt with an empty identity (avatar has
    # not yet learned anything), matching the first-turn state of the bug report.
    system_prompt = (
        DynamicPromptBuilder()
        .build_prompt(assistant_name="", assistant_identity=[], system_time="now")
        .messages[0]
        .content
    )

    from langchain_core.messages import SystemMessage

    response = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=STORY)]
    )

    identity_calls = [
        tc
        for tc in (response.tool_calls or [])
        if tc["name"] == "update_self_identity_mem_from_user_txt"
    ]

    print(f"\n=== total tool calls: {len(response.tool_calls or [])} ===")
    print(f"=== update_self_identity_mem_from_user_txt calls: {len(identity_calls)} ===\n")
    for i, tc in enumerate(identity_calls, 1):
        print(f"{i}. fact: {tc['args'].get('assistant_fact')!r}")
        print(f"   context: {tc['args'].get('fact_context')!r}")

    assert len(identity_calls) >= 5, (
        f"Expected the avatar to learn multiple distinct facts (>=5), "
        f"got {len(identity_calls)}. The story summarization bug is NOT fixed."
    )
    print("\nPASS (phase 1): avatar extracted multiple distinct facts (one tool call each).")

    # --- Phase 2: can the avatar recount the story precisely from retrieved facts? ---
    # load_consciousness loads every identity_memory doc into the system prompt;
    # we simulate that by feeding the just-learned facts back as identity docs.
    from langchain_core.documents import Document

    learned_docs = [
        Document(
            page_content=wrap_fact_with_context(
                tc["args"]["assistant_fact"], tc["args"].get("fact_context", "")
            ),
            metadata={"fact": tc["args"]["assistant_fact"]},
        )
        for tc in identity_calls
    ]
    print("\n=== example stored page_content (wrapped with full context) ===")
    print(learned_docs[1].page_content if len(learned_docs) > 1 else learned_docs[0].page_content)

    # Every stored fact must carry the FACT_CONTEXT wrapper preserving the original story.
    assert all(
        d.page_content.startswith("<FACT_CONTEXT_AND_FACT>")
        and "<FACT_CONTEXT>" in d.page_content
        and "<FACT>" in d.page_content
        for d in learned_docs
    ), "Stored facts are not wrapped with their original background context."

    retell_prompt = (
        DynamicPromptBuilder()
        .build_prompt(
            assistant_name="",
            assistant_identity=learned_docs,
            system_time="now",
        )
        .messages[0]
        .content
    )

    # Plain chat model (no tools) for the retell turn.
    chat_model = init_model(tools=[], tool_choice="auto")
    retell = await chat_model.ainvoke(
        [
            SystemMessage(content=retell_prompt),
            HumanMessage(content="Tell me the story about the day you went to get your glasses."),
        ]
    )
    retell_text = retell.content if isinstance(retell.content, str) else str(retell.content)
    print("\n=== avatar retell ===\n" + retell_text + "\n")

    expected_specifics = [
        "Crouching Tiger",
        "Walmart",
        "glasses",
        "mom",
    ]
    missing = [s for s in expected_specifics if s.lower() not in retell_text.lower()]
    assert not missing, f"Retell missing precise details: {missing}"
    print("PASS (phase 2): avatar recounted the story with precise retrieved details.")

    # --- Phase 3: same story told as the USER's own → learn_information_about_the_user ---
    user_self_story = (
        "Let me tell you about myself. I am a man. The day I was going for "
        'glasses, I picked up my glasses before I saw the movie "Crouching Tiger, '
        'Hidden Dragon". I thought it was good I got my glasses before the movie '
        "because everything was now clear. I could read the signs in the back of "
        "the Walmart and I could see the individual leaves on the trees. I loved "
        'watching action movies and science fiction movies with my mom. I was her '
        '"guy" and her "buddy".'
    )

    user_response = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_self_story)]
    )
    user_calls = [
        tc
        for tc in (user_response.tool_calls or [])
        if tc["name"] == "learn_information_about_the_user"
    ]
    print(f"\n=== learn_information_about_the_user calls: {len(user_calls)} ===")
    for i, tc in enumerate(user_calls, 1):
        print(f"{i}. fact: {tc['args'].get('user_fact')!r}")

    assert len(user_calls) >= 5, (
        f"Expected the avatar to learn multiple distinct USER facts (>=5), "
        f"got {len(user_calls)}."
    )
    print("\nPASS (phase 3): avatar extracted multiple distinct USER facts (one call each).")


if __name__ == "__main__":
    asyncio.run(main())
