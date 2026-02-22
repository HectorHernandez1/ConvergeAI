import json
from typing import Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from models import SolverResponse
from config import settings
from utils.cache import get_cache_key, get_cached_response, cache_response
from utils.image_types import ExtractedImage
from .base_solver import BaseSolver
from utils.token_counter import count_openai_tokens

class OpenAISolver(BaseSolver):
    def __init__(self):
        super().__init__("OpenAI " + settings.openai_model)
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
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

        # Build user message content (vision-aware)
        if images:
            user_content = self._build_vision_content(prompt, images)
        else:
            user_content = prompt

        # Determine which parameter to use for token limits
        # GPT-5.x models use max_completion_tokens
        # GPT-4.x models (including gpt-4.1-nano, gpt-4o, gpt-4-turbo) use max_tokens
        uses_new_api = "gpt-5" in self.model

        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert academic assistant. Always respond with valid JSON only, no markdown fences."},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3 if iteration == 1 else 0.5,
        }

        # Add the appropriate token limit parameter based on model version
        if uses_new_api:
            request_params["max_completion_tokens"] = settings.max_tokens
        else:
            request_params["max_tokens"] = settings.max_tokens

        response = await self.client.chat.completions.create(**request_params)
        
        output_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = input_tokens + output_tokens
        cost = self.estimate_cost(input_tokens, output_tokens)
        
        response_data = self._parse_response(output_text)
        normalized_answers = self._normalize_answers(response_data["answers"])
        solver_response = SolverResponse(
            model_name=self.model_name,
            iteration=iteration,
            answers=normalized_answers,
            tokens_used=total_tokens,
            cost_usd=cost
        )
        
        if use_cache:
            cache_response(cache_key, solver_response.model_dump_json())

        return solver_response
    
    def _build_vision_content(self, prompt: str, images: list[ExtractedImage]) -> list[dict]:
        """Build OpenAI vision-format content array with text + images."""
        image_index = "\n\nThe following images are attached:\n"
        for i, img in enumerate(images, 1):
            image_index += f"- Image {i}: {img.label}\n"
        image_index += "\nRefer to these images when answering questions that involve charts, graphs, figures, or visual data.\n"

        content = [{"type": "text", "text": prompt + image_index}]
        for img in images:
            data_url = f"data:{img.media_type};base64,{img.to_base64()}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "auto"}
            })
        return content

    def count_tokens(self, text: str) -> int:
        return count_openai_tokens(text)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = {
            "gpt-5.2": {"input": 0.00175, "output": 0.014},
            "gpt-5.1": {"input": 0.00125, "output": 0.01},
            "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
        }
        rates = pricing.get(self.model, {"input": 0.0025, "output": 0.01})
        return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])
    
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
                disagreement_summary=previous_answers["disagreement_summary"]
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
            raise ValueError("Invalid JSON response from model")
