import json
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from models import SolverResponse
from config import settings
from utils.cache import get_cache_key, get_cached_response, cache_response
from .base_solver import BaseSolver

class AnthropicSolver(BaseSolver):
    def __init__(self):
        super().__init__("Anthropic " + settings.anthropic_model)
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def solve(self, problem: str, references: str = None, 
                    previous_answers: dict = None, iteration: int = 1) -> SolverResponse:
        prompt = self._build_prompt(problem, references, previous_answers, iteration)
        
        if settings.enable_cache:
            cache_key = get_cache_key(self.model_name, prompt)
            cached = get_cached_response(cache_key)
            if cached:
                return SolverResponse(**json.loads(cached))
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=settings.max_tokens,
            temperature=0.3 if iteration == 1 else 0.5,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are an expert academic assistant. Always respond with valid JSON only, no markdown fences."
        )
        
        output_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
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
        
        if settings.enable_cache:
            cache_response(cache_key, solver_response.model_dump_json())
        
        return solver_response
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = {
            "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
            "claude-haiku-4-5": {"input": 0.001, "output": 0.005},
            "claude-opus-4-5": {"input": 0.005, "output": 0.025},
        }
        rates = pricing.get(self.model, {"input": 0.003, "output": 0.015})
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
        except json.JSONDecodeError as e:
            if "```json" in output:
                json_start = output.find("```json") + 7
                json_end = output.find("```", json_start)
                if json_end > json_start:
                    extracted = output[json_start:json_end].strip()
                    try:
                        data = json.loads(extracted)
                        if isinstance(data, list):
                            return {"answers": data}
                        if "answers" in data:
                            return data
                    except json.JSONDecodeError:
                        pass
            raise ValueError("Invalid JSON response from model")
