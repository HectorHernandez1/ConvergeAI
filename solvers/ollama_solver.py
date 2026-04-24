import json
from typing import Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from models import SolverResponse
from config import settings
from utils.cache import get_cache_key, get_cached_response, cache_response
from utils.image_types import ExtractedImage
from .base_solver import BaseSolver

# Fallback prefixes used only if /api/show is unreachable
_VISION_PREFIX_FALLBACK = (
    "gemma3", "gemma4",
    "llava", "bakllava",
    "llama3.2-vision", "llama4",
    "minicpm-v",
    "qwen2.5vl", "qwen2-vl", "qwen3-vl", "qwen3vl",
    "moondream", "pixtral",
)


def _repair_truncated_json(s: str) -> Optional[object]:
    """Try to salvage JSON that got cut off mid-output by trimming to the last
    complete element and closing open brackets/braces. Returns None if no fix
    works."""
    s = s.rstrip().rstrip(",")
    # Walk the prefix tracking brace/bracket depth and string state, find the
    # last position where we are outside any string and could close cleanly.
    depth_stack = []
    in_str = False
    escape = False
    last_complete = -1
    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c in "{[":
            depth_stack.append(c)
        elif c in "}]":
            if depth_stack:
                depth_stack.pop()
            if not depth_stack:
                last_complete = i
    # If there's a complete top-level object, just use that.
    if not depth_stack and last_complete >= 0:
        try:
            return json.loads(s[: last_complete + 1])
        except json.JSONDecodeError:
            pass
    # Otherwise, trim back to the last comma at depth==1 (end of a complete
    # answer entry) and close what remains.
    depth = 0
    in_str = False
    escape = False
    trim_at = -1
    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
        elif c == "," and depth == 2:
            trim_at = i
    if trim_at > 0:
        candidate = s[:trim_at]
        # Close any still-open structures.
        stack = []
        in_str2 = False
        esc2 = False
        for c in candidate:
            if esc2:
                esc2 = False
                continue
            if c == "\\" and in_str2:
                esc2 = True
                continue
            if c == '"':
                in_str2 = not in_str2
                continue
            if in_str2:
                continue
            if c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack:
                    stack.pop()
        closers = "".join("}" if o == "{" else "]" for o in reversed(stack))
        try:
            return json.loads(candidate + closers)
        except json.JSONDecodeError:
            return None
    return None


def _probe_vision(base_url: str, model: str) -> Optional[bool]:
    """Ask the Ollama server whether this model declares the 'vision' capability.
    Returns None on network/parse failure so the caller can fall back."""
    try:
        resp = httpx.post(f"{base_url}/api/show", json={"model": model}, timeout=10.0)
        resp.raise_for_status()
        caps = resp.json().get("capabilities") or []
        return "vision" in caps
    except Exception:
        return None


class OllamaSolver(BaseSolver):
    def __init__(self, model: str, short_name: Optional[str] = None):
        super().__init__(f"Ollama {model}")
        self.model = model
        self.short_name = short_name or f"Ollama-{model}"
        self.base_url = settings.ollama_base_url.rstrip("/")

        probed = _probe_vision(self.base_url, model)
        if probed is None:
            name = model.lower()
            self.supports_vision = any(name.startswith(p) for p in _VISION_PREFIX_FALLBACK)
        else:
            self.supports_vision = probed

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
            # Disable thinking for thinking-capable models (qwen3.x, gemma4, etc).
            # With thinking on, reasoning tokens consume num_predict and
            # message.content can come back empty, breaking JSON parsing.
            "think": False,
            "options": {
                "temperature": 0.3 if iteration == 1 else 0.5,
                "num_predict": settings.ollama_max_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }

        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        output_text = data["message"]["content"]
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        done_reason = data.get("done_reason")
        total_tokens = input_tokens + output_tokens
        cost = self.estimate_cost(input_tokens, output_tokens)

        response_data = self._parse_response(output_text, done_reason=done_reason)
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

    def _parse_response(self, output: str, done_reason: Optional[str] = None) -> dict:
        stripped = (output or "").strip()
        if not stripped:
            raise ValueError(
                f"Ollama model {self.model} returned empty content "
                f"(done_reason={done_reason}). This usually means thinking tokens "
                "consumed num_predict — confirm 'think': false is set."
            )
        try:
            data = json.loads(stripped)
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

            # Last-ditch: if the response was truncated mid-JSON (done_reason=length
            # or closing tokens missing), try to repair by closing open braces/brackets.
            # This salvages partial runs instead of wasting a 20-min iteration.
            if done_reason == "length" or not stripped.rstrip().endswith(("}", "]")):
                repaired = _repair_truncated_json(stripped)
                if repaired is not None:
                    if isinstance(repaired, list):
                        return {"answers": repaired}
                    if "answers" in repaired:
                        return repaired

            head = stripped[:200].replace("\n", " ")
            tail = stripped[-200:].replace("\n", " ")
            raise ValueError(
                f"Invalid JSON response from Ollama model {self.model} "
                f"(done_reason={done_reason}, length={len(stripped)}). "
                f"Head: {head!r} ... Tail: {tail!r}"
            )
