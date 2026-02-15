"""Centralized LLM model defaults and configuration.

Single source of truth for all model names and cost estimates
used across the codebase. Change model defaults here only.
"""

# Default model per provider (used when env vars are not set)
DEFAULT_MODELS = {
    "oca": "oca/gpt5.2",
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "ollama": "llama3.1",
    "lm_studio": "local-model",
    "oci_genai": "cohere.command-r-plus",
}

# OCA fallback chain (in order of preference)
OCA_FALLBACK_MODELS = [
    "oca/gpt5.2",
    "oca/gpt5",
    "oca/gpt-oss-120b",
    "oca/llama4",
    "oca/openai-o3",
]

# Token cost estimates per 1K tokens (approximate, USD)
TOKEN_COSTS = {
    "oca/gpt5.2": {"input": 0.001, "output": 0.002},
    "oca/gpt5": {"input": 0.001, "output": 0.002},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}
