# src/browspi/ui/llm_utils.py

from langchain_core.language_models import BaseLanguageModel


def get_llm_provider(provider: str, api_key: str | None = None) -> BaseLanguageModel:
    """
    Khởi tạo và trả về một đối tượng LLM dựa trên nhà cung cấp được chọn.
    """
    if provider == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI

            if not api_key:
                raise ValueError("API Key is required for OpenAI")
            return ChatOpenAI(
                model="gpt-4o", api_key=api_key, temperature=0, max_tokens=1500
            )
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'langchain-openai' to be installed. Please run 'poetry add langchain-openai'"
            )

    elif provider == "Mistral":
        try:
            from langchain_mistralai import ChatMistralAI

            if not api_key:
                raise ValueError("API Key is required for Mistral")
            return ChatMistralAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Mistral provider requires 'langchain-mistralai' to be installed. It seems to be installed already."
            )

    # Thêm các nhà cung cấp khác ở đây nếu cần
    # elif provider == "Google":
    #     ...

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
