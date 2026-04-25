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
    openai_model: str = "gpt-5.2"

    # Anthropic models available:
    # claude-sonnet-4-5 ($0.003/input 1k Tok, $0.015/output 1k Tok),
    # claude-haiku-4-5 ($0.001/input 1k Tok, $0.005/output 1k Tok),
    # claude-opus-4-5 ($0.005/input 1k Tok, $0.025/output 1k Tok)
    anthropic_model: str = "claude-sonnet-4-6"

    # Ollama (local open-source models via Ollama runtime)
    # Run `ollama list` to see installed models.
    # Examples: "gemma4:31b" (vision), "qwen3.5:35b-a3b", "deepseek-r1:32b".
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 1800.0  # 30 min — large local models can be slow
    # Ollama output cap. 8k (the cloud default) truncates multi-answer JSON
    # mid-stream on local models that echo question_text. 20k handled all
    # 8 answers cleanly in test runs; raising further (e.g. 32k) gave qwen3.x
    # enough rope to spend tokens on increasingly verbose reasoning and still
    # truncate, so we cap it back at 20k.
    ollama_max_tokens: int = 20000
    # Ollama input context window. Ollama's default is 4096 — far smaller than
    # modern model capacities (262k for qwen3.x / gemma4) — which silently
    # truncates prompts with references + many images. 32768 gives comfortable
    # headroom for quiz + references + ~40 images. Empirically, raising
    # further (e.g. 64k) lets verbose models pull in more reference text and
    # write longer reasoning per answer, which then truncates output anyway —
    # 32k is the sweet spot for this workload.
    ollama_num_ctx: int = 32768

    # Solver pair used for consensus. Each entry is either "openai", "anthropic",
    # or "ollama:<model>" (e.g. "ollama:gemma4:31b"). CLI --solvers overrides.
    # Default: cloud pair. Override for local with e.g.:
    #   --solvers ollama:gemma4:31b,ollama:qwen3.6:35b
    solvers: list[str] = ["openai", "anthropic"]

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

    max_images: int = 20

    similarity_threshold: float = 0.85
    numerical_tolerance: float = 0.01

settings = Settings()
