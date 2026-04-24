import json
from abc import ABC, abstractmethod
from typing import Optional
from models import SolverResponse, Answer
from utils.image_types import ExtractedImage

class BaseSolver(ABC):
    short_name: str = ""

    def __init__(self, model_name: str):
        self.model_name = model_name
        if not self.short_name:
            self.short_name = model_name

    @abstractmethod
    async def solve(self, problem: str, references: str = None,
                    previous_answers: dict = None, iteration: int = 1,
                    images: Optional[list[ExtractedImage]] = None) -> SolverResponse:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pass

    def _normalize_answers(self, answers: list) -> list[Answer]:
        """
        Normalize answer data from various model output formats into Answer objects.

        Handles:
        - Answers as dicts with string or non-string 'answer' fields
        - Answers as lists, dicts, numbers, or other non-string types
        - Already-constructed Answer objects
        - Missing or malformed fields with sensible defaults
        """
        normalized = []
        for idx, ans in enumerate(answers):
            try:
                if isinstance(ans, dict):
                    ans_dict = ans.copy()

                    # Convert non-string answer to JSON string
                    answer_val = ans_dict.get("answer")
                    if answer_val is not None and not isinstance(answer_val, str):
                        ans_dict["answer"] = json.dumps(answer_val, ensure_ascii=False)
                    elif answer_val is None:
                        ans_dict["answer"] = ""

                    # Ensure required fields have defaults
                    ans_dict.setdefault("question_number", idx + 1)
                    ans_dict.setdefault("question_text", "")
                    ans_dict.setdefault("reasoning", "")
                    ans_dict.setdefault("confidence", "medium")
                    ans_dict.setdefault("references_cited", [])

                    normalized.append(Answer(**ans_dict))

                elif isinstance(ans, Answer):
                    # Use model_copy to avoid mutating the original
                    if not isinstance(ans.answer, str):
                        ans = ans.model_copy(update={
                            "answer": json.dumps(ans.answer, ensure_ascii=False)
                        })
                    normalized.append(ans)
                else:
                    # Handle unexpected types by converting to string
                    normalized.append(Answer(
                        question_number=idx + 1,
                        question_text="",
                        answer=str(ans) if ans is not None else "",
                        reasoning="",
                        confidence="low",
                        references_cited=[]
                    ))
            except Exception as e:
                # If normalization fails for one answer, create a placeholder
                normalized.append(Answer(
                    question_number=idx + 1,
                    question_text="",
                    answer=f"Error normalizing answer: {str(e)}",
                    reasoning="",
                    confidence="low",
                    references_cited=[]
                ))
        return normalized
