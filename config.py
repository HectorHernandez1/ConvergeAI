import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load ONLY API keys from .env file
load_dotenv()

class Settings(BaseModel):
    # API keys - loaded from .env file only
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # All other settings - hardcoded defaults, edit this file to change
    # OpenAI models available:
    # gpt-5.2 ($0.00175/input 1k Tok, $0.014/output 1k Tok),
    # gpt-5.1 ($0.00125/input 1k Tok, $0.01/output 1k Tok),
    # gpt-4.1-nano ($0.0001/input 1k Tok, $0.0004/output 1k Tok)
    openai_model: str = "gpt-4.1-nano"

    # Anthropic models available:
    # claude-sonnet-4-5 ($0.003/input 1k Tok, $0.015/output 1k Tok),
    # claude-haiku-4-5 ($0.001/input 1k Tok, $0.005/output 1k Tok),
    # claude-opus-4-5 ($0.005/input 1k Tok, $0.025/output 1k Tok)
    anthropic_model: str = "claude-haiku-4-5"

    # Max tokens for model responses
    max_tokens: int = 8000
    max_iterations: int = 5
    agreement_threshold: float = 1.0
    early_stop_threshold: float = 0.90  # Stop at 90% agreement instead of 100%

    max_cost_usd: float = 5.0

    input_dir: str = "input"
    output_dir: str = "output"
    log_dir: str = "logs"

    enable_cache: bool = False
    cache_ttl_hours: int = 24

    similarity_threshold: float = 0.85
    numerical_tolerance: float = 0.01

settings = Settings()
