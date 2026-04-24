import json
from typing import Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from models import SolverResponse
from config import settings
from utils.cache import get_cache_key, get_cached_response, cache_response
from utils.image_types import ExtractedImage
from .base_solver import BaseSolver

# Known vision-capable Ollama model prefixes
VISION_MODEL_PREFIXES = (
    "gemma3", "gemma4",
    "llava", "bakllava",
    "llama3.2-vision", "llama4",
    "minicpm-v",
    "qwen2.5vl", "qwen2-vl", "qwen3-vl", "qwen3vl",
    "moondream", "pixtral",
)


def _model_supports_vision(model: str) -> bool:
    name = model.lower()
    return any(name.startswith(prefix) for prefix in VISION_MODEL_PREFIXES)


class OllamaSolver(BaseSolver):
    def __init__(self, model: str, short_name: Optional[str] = None):
        super().__init__(f"Ollama {model}")
        self.model = model
        self.short_name = short_name or f"Ollama-{model}"
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.supports_vision = _model_supports_vision(model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def solve(self, problem: str, references: str = None,
                    previous_answers: dict = None, iteration: int = 1,
                    images: Optional[list[ExtractedImage]] = None) -> SolverResponse:
        prompt = self._build_prompt(problem, references, previous_answers, iteration)

        use_cache = settings.enable_cache and not images
        if use_cache:
            cache_key = get_cache_key(self.model_name, prompt)
            cached = get_cached_response(cache_key)
            if cached:
                return SolverResponse(**json.loads(cached))

        user_message = {"role": "user", "content": prompt}
        if images and self.supports_vision:
            user_message["images"] = [img.to_base64() for img in images]
            image_index = "\n\nThe following images are attached:\n"
            for i, img in enumerate(images, 1):
                image_index += f"- Image {i}: {img.label}\n"
            image_index += "\nRefer to these images when answering questions that involve charts, graphs, figures, or visual data.\n"
            user_message["content"] = prompt + image_index

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert academic assistant. Always respond with valid JSON only, no markdown fences."},
                user_message,
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3 if iteration == 1 else 0.5,
                "num_predict": settings.max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        output_text = data["message"]["content"]
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        total_tokens = input_tokens + output_tokens
        cost = self.estimate_cost(input_tokens, output_tokens)

        response_data = self._parse_response(output_text)
        normalized_answers = self._normalize_answers(response_data["answers"])
        solver_response = SolverResponse(
            model_name=self.model_name,
            iteration=iteration,
            answers=normalized_answers,
            tokens_used=total_tokens,
            cost_usd=cost,
        )

        if use_cache:
            cache_response(cache_key, solver_response.model_dump_json())

        return solver_response

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    def _build_prompt(self, problem: str, references: str, previous_answers: dict, iteration: int) -> str:
        if iteration == 1:
            with open("prompts/initial_solve.txt", "r") as f:
                template = f.read()
            references_text = references or "No reference materials provided."
            return template.format(problem_text=problem, references_text=references_text)
        else:
            with open("prompts/refinement.txt", "r") as f:
                template = f.read()
            return template.format(
                problem_text=problem,
                your_previous_answers=json.dumps(previous_answers["your_answers"], indent=2),
                other_model_answers=json.dumps(previous_answers["other_answers"], indent=2),
                disagreement_summary=previous_answers["disagreement_summary"],
            )

    def _parse_response(self, output: str) -> dict:
        try:
            data = json.loads(output.strip())
            if isinstance(data, list):
                return {"answers": data}
            if "answers" not in data:
                raise ValueError("Missing 'answers' field")
            return data
        except json.JSONDecodeError:
            if "```json" in output:
                json_start = output.find("```json") + 7
                json_end = output.find("```", json_start)
                if json_end > json_start:
                    data = json.loads(output[json_start:json_end].strip())
                    if isinstance(data, list):
                        return {"answers": data}
                    if "answers" in data:
                        return data
            raise ValueError("Invalid JSON response from Ollama model")
