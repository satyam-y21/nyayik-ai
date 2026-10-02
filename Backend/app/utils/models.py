from Backend.app.config import settings

_llm_cache = {}


def get_llm():
    """Returns cached LLM instance, creates one if needed."""
    provider = settings.LLM_PROVIDER.lower()
    model_name = settings.MODEL_NAME
    cache_key = f"{provider}:{model_name}"

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    print(f"Initializing LLM: {provider} ({model_name})")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model_name, temperature=0)
    elif provider == "gemini":
        import os

        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=0
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(model=model_name, temperature=0)
    elif provider == "groq":
        import os

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    _llm_cache[cache_key] = llm
    return llm
