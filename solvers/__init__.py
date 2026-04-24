from .base_solver import BaseSolver
from .openai_solver import OpenAISolver
from .anthropic_solver import AnthropicSolver
from .ollama_solver import OllamaSolver


def build_solver(spec: str) -> BaseSolver:
    """
    Build a solver from a string spec.

    Specs:
      - "openai"                   → OpenAISolver (uses settings.openai_model)
      - "anthropic"                → AnthropicSolver (uses settings.anthropic_model)
      - "ollama:<model>"           → OllamaSolver(<model>)
        e.g. "ollama:gemma4:31b", "ollama:qwen3.5:35b-a3b"
    """
    spec = spec.strip()
    if spec == "openai":
        return OpenAISolver()
    if spec == "anthropic":
        return AnthropicSolver()
    if spec.startswith("ollama:"):
        model = spec[len("ollama:"):]
        if not model:
            raise ValueError(f"Ollama spec is missing model name: {spec!r}")
        return OllamaSolver(model)
    raise ValueError(
        f"Unknown solver spec {spec!r}. Expected 'openai', 'anthropic', or 'ollama:<model>'."
    )
