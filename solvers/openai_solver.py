import asyncio
import json
from typing import Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from models import SolverResponse, Answer
from config import settings
from utils.cache import get_cache_key, get_cached_response, cache_response
from .base_solver import BaseSolver
from utils.token_counter import count_openai_tokens

class OpenAISolver(BaseSolver):
    def __init__(self):
        super().__init__("OpenAI " + settings.openai_model)
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def solve(self, problem: str, references: str = None, 
                    previous_answers: dict = None, iteration: int = 1) -> SolverResponse:
        prompt = self._build_prompt(problem, references, previous_answers, iteration)
        
        if settings.enable_cache:
            cache_key = get_cache_key(self.model_name, prompt)
            cached = get_cached_response(cache_key)
            if cached:
                return SolverResponse(**json.loads(cached))
        
        input_tokens = self.count_tokens(prompt)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert academic assistant. Always respond with valid JSON only, no markdown fences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 if iteration == 1 else 0.5,
            max_tokens=4000
        )
        
        output_text = response.choices[0].message.content
        output_tokens = self.count_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        cost = self.estimate_cost(input_tokens, output_tokens)
        
        response_data = self._parse_response(output_text)
        solver_response = SolverResponse(
            model_name=self.model_name,
            iteration=iteration,
            answers=response_data["answers"],
            tokens_used=total_tokens,
            cost_usd=cost
        )
        
        if settings.enable_cache:
            cache_response(cache_key, solver_response.model_dump_json())
        
        return solver_response
    
    def count_tokens(self, text: str) -> int:
        return count_openai_tokens(text)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
        }
        rates = pricing.get(self.model, {"input": 0.005, "output": 0.015})
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
