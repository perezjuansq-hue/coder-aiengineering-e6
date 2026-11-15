"""Configuración del LLM por .env (misma convención que las entregas anteriores)."""
import logging
import os
from enum import Enum

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger("llm")


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


MODELOS_POR_DEFECTO = {
    Provider.OPENAI: "gpt-6-luna",
    Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
}


def crear_modelo(max_tokens: int = 2048) -> BaseChatModel:
    load_dotenv()
    valor_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not valor_provider:
        raise ValueError("Falta LLM_PROVIDER (valores aceptados: openai, anthropic)")
    provider = Provider(valor_provider)

    if provider == Provider.OPENAI:
        modelo = os.getenv("OPENAI_MODEL") or MODELOS_POR_DEFECTO[provider]
        logger.info("LLM: openai | %s", modelo)
        return ChatOpenAI(model=modelo, max_tokens=max_tokens)

    modelo = os.getenv("ANTHROPIC_MODEL") or MODELOS_POR_DEFECTO[provider]
    logger.info("LLM: anthropic | %s", modelo)
    return ChatAnthropic(model=modelo, max_tokens=max_tokens)


def nombre_modelo(modelo: BaseChatModel) -> str:
    return getattr(modelo, "model_name", None) or getattr(modelo, "model", "?")
