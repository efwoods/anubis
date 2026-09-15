"""Swapping a basemodel is changing the existing provider env knobs.

Text inference reads MODEL / LLM_PROVIDER_API_KEY / LLM_PROVIDER_BASE_URL.
Image description reads IMAGE_MODEL / IMAGE_MODEL_API_KEY / IMAGE_MODEL_BASE_URL.
NVIDIA's hosted catalog is OpenAI-compatible, so both text and image clients
are ChatOpenAI pointed at LLM_PROVIDER_* / IMAGE_MODEL_*.
"""

from src.anubis.utils import model as model_module


class _FakeChatOpenAI:
    def __init__(self, **keyword_arguments):
        self.keyword_arguments = keyword_arguments


def test_nvidia_unbound_client_uses_llm_provider_base_url_and_model(monkeypatch):
    captured = {}

    class CapturingChatOpenAI(_FakeChatOpenAI):
        def __init__(self, **keyword_arguments):
            captured.update(keyword_arguments)
            super().__init__(**keyword_arguments)

    monkeypatch.setenv("MODEL_PROVIDER", "NVIDIA")
    monkeypatch.setenv("MODEL", "meta/llama-3.2-11b-vision-instruct")
    monkeypatch.setenv("LLM_PROVIDER_API_KEY", "nvapi-test")
    monkeypatch.setenv(
        "LLM_PROVIDER_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    monkeypatch.setattr("langchain_openai.ChatOpenAI", CapturingChatOpenAI)

    client = model_module.init_chat_model_unbound()
    assert captured["model"] == "meta/llama-3.2-11b-vision-instruct"
    assert captured["api_key"] == "nvapi-test"
    assert captured["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert captured["stream_usage"] is False
    assert client.keyword_arguments == captured


def test_image_description_client_uses_image_model_knobs(monkeypatch):
    captured = {}

    class CapturingChatOpenAI(_FakeChatOpenAI):
        def __init__(self, **keyword_arguments):
            captured.update(keyword_arguments)
            super().__init__(**keyword_arguments)

    monkeypatch.setenv("MODEL_PROVIDER", "NVIDIA")
    monkeypatch.setenv("IMAGE_MODEL", "meta/llama-3.2-11b-vision-instruct")
    monkeypatch.setenv("IMAGE_MODEL_API_KEY", "nvapi-test")
    monkeypatch.setenv(
        "IMAGE_MODEL_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    monkeypatch.setattr("langchain_openai.ChatOpenAI", CapturingChatOpenAI)

    client = model_module.init_image_description_model()
    assert captured["model"] == "meta/llama-3.2-11b-vision-instruct"
    assert captured["api_key"] == "nvapi-test"
    assert captured["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert client.keyword_arguments == captured


def test_nvidia_unbound_client_is_not_wrapped_when_already_on_nim(monkeypatch):
    captured = {}

    class CapturingChatOpenAI(_FakeChatOpenAI):
        def __init__(self, **keyword_arguments):
            captured.update(keyword_arguments)
            super().__init__(**keyword_arguments)

    monkeypatch.setenv("DEV", "TRUE")
    monkeypatch.setenv("MODEL_PROVIDER", "NVIDIA")
    monkeypatch.setenv("MODEL", "meta/llama-3.2-11b-vision-instruct")
    monkeypatch.setenv("LLM_PROVIDER_API_KEY", "nvapi-test")
    monkeypatch.setenv(
        "LLM_PROVIDER_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    monkeypatch.setattr("langchain_openai.ChatOpenAI", CapturingChatOpenAI)

    client = model_module.init_chat_model_unbound()
    assert isinstance(client, CapturingChatOpenAI)
    assert captured["model"] == "meta/llama-3.2-11b-vision-instruct"


def test_open_ai_unbound_client_wraps_nvidia_nim_credit_fallback_in_dev(monkeypatch):
    monkeypatch.setenv("DEV", "TRUE")
    monkeypatch.setenv("MODEL_PROVIDER", "OPEN_AI")
    monkeypatch.setenv("MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("LLM_PROVIDER_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("NVIDIA_NIM_MODEL", "meta/llama-3.2-90b-vision-instruct")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeChatOpenAI)

    client = model_module.init_chat_model_unbound()
    assert isinstance(client, model_module.NvidiaNimCreditFallbackChatModel)


def test_credit_exhaustion_retries_on_nvidia_nim(monkeypatch):
    from langchain_core.messages import AIMessage

    monkeypatch.setenv("DEV", "TRUE")
    monkeypatch.setenv("MODEL_PROVIDER", "OPEN_AI")
    monkeypatch.setenv("MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_NIM_MODEL", "meta/llama-3.2-90b-vision-instruct")
    from src.anubis.utils.context import GlobalContext

    class ExhaustedPrimary:
        def invoke(self, *_arguments, **_keyword_arguments):
            raise RuntimeError("Error code: 429 - insufficient_quota")

        def bind_tools(self, *_arguments, **_keyword_arguments):
            return self

    class NvidiaNimFallback:
        def invoke(self, *_arguments, **_keyword_arguments):
            return AIMessage(content="from nvidia nim")

        def bind_tools(self, *_arguments, **_keyword_arguments):
            return self

    context = GlobalContext()
    wrapped = model_module.NvidiaNimCreditFallbackChatModel(
        primary=ExhaustedPrimary(),
        fallback=NvidiaNimFallback(),
        context=context,
    )
    reply = wrapped.invoke([{"role": "user", "content": "hello"}])
    assert reply.content == "from nvidia nim"
    record = model_module.text_inference_record(context)
    assert record["text_model"] == "meta/llama-3.2-90b-vision-instruct"
    assert record["text_model_provider"] == "NVIDIA"
    assert record["text_model_credit_fallback"] is True


def test_vendor_credit_is_exhausted_matches_known_refusals():
    assert model_module.vendor_credit_is_exhausted(
        RuntimeError("insufficient_quota")
    )
    assert not model_module.vendor_credit_is_exhausted(RuntimeError("rate limited"))


def test_vendor_key_is_refused_is_not_empty_funds():
    key_error = RuntimeError(
        "ElevenLabs rejected the request (401 invalid_api_key: Invalid API key)"
    )
    assert model_module.vendor_key_is_refused(key_error)
    assert not model_module.vendor_credit_is_exhausted(key_error)
    quota_error = RuntimeError("Error code: 429 - insufficient_quota")
    assert model_module.vendor_credit_is_exhausted(quota_error)
    assert not model_module.vendor_key_is_refused(quota_error)


def test_nvidia_90b_hosted_input_limit_is_32768_even_when_model_token_limit_is_larger(
    monkeypatch,
):
    monkeypatch.setenv("MODEL_PROVIDER", "NVIDIA")
    monkeypatch.setenv("MODEL", "meta/llama-3.2-90b-vision-instruct")
    monkeypatch.setenv("MODEL_TOKEN_LIMIT", "400000")
    from src.anubis.utils.context import GlobalContext

    context = GlobalContext()
    assert model_module.hosted_inference_input_token_limit(context) == 32768


def test_nvidia_11b_keeps_configured_model_token_limit(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "NVIDIA")
    monkeypatch.setenv("MODEL", "meta/llama-3.2-11b-vision-instruct")
    monkeypatch.setenv("MODEL_TOKEN_LIMIT", "400000")
    from src.anubis.utils.context import GlobalContext

    context = GlobalContext()
    assert model_module.hosted_inference_input_token_limit(context) == 400000


def test_extra_tools_are_kept_on_the_nvidia_90b_window():
    from types import SimpleNamespace

    from src.anubis.utils.context_compression import extra_tools_within_inference_window

    huge = SimpleNamespace(name="query_platform_metrics", description="x" * 8000, args={})
    core = SimpleNamespace(name="recall_memories", description="core", args={})
    kept = extra_tools_within_inference_window(
        [huge],
        core_tools=[core],
        system_text="y" * 20000,
        messages=[],
        window=32768,
    )
    assert kept == [huge]


def test_extra_tools_are_kept_when_the_inference_window_is_larger_than_the_nvidia_90b_cap():
    from types import SimpleNamespace

    from src.anubis.utils.context_compression import extra_tools_within_inference_window

    huge = SimpleNamespace(name="query_platform_metrics", description="x" * 8000, args={})
    kept = extra_tools_within_inference_window(
        [huge],
        core_tools=[],
        system_text="y" * 20000,
        messages=[],
        window=400000,
    )
    assert kept == [huge]
