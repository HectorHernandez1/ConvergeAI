from abc import ABC, abstractmethod
from models import SolverResponse

class BaseSolver(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    async def solve(self, problem: str, references: str = None, 
                    previous_answers: dict = None, iteration: int = 1) -> SolverResponse:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass
