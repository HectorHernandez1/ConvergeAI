import re
import unicodedata
from difflib import SequenceMatcher
from typing import Tuple, Optional
from models import SolverResponse, ComparisonResult, Answer
from config import settings

# Map Unicode subscript/superscript characters to ASCII equivalents
_UNICODE_TO_ASCII = str.maketrans({
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')',
    'ₐ': 'a', 'ₑ': 'e', 'ₒ': 'o', 'ₓ': 'x', 'ₕ': 'h',
    'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₚ': 'p',
    'ₛ': 's', 'ₜ': 't',
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
    'ᵃ': 'a', 'ᵇ': 'b', 'ᵈ': 'd', 'ᵉ': 'e', 'ᶠ': 'f',
    'ᵍ': 'g', 'ⁱ': 'i', 'ʲ': 'j', 'ᵏ': 'k', 'ˡ': 'l',
    'ᵐ': 'm', 'ⁿ': 'n', 'ᵒ': 'o', 'ᵖ': 'p', 'ʳ': 'r',
    'ˢ': 's', 'ᵗ': 't', 'ᵘ': 'u', 'ᵛ': 'v', 'ʷ': 'w',
    'ˣ': 'x', 'ʸ': 'y', 'ᶻ': 'z',
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'σ': 'sigma', 'μ': 'mu', 'η': 'eta', 'θ': 'theta',
    'λ': 'lambda', 'π': 'pi', 'ρ': 'rho', 'τ': 'tau',
    'φ': 'phi', 'ψ': 'psi', 'ω': 'omega',
})

def compare_answers(response_a: SolverResponse, 
                    response_b: SolverResponse) -> ComparisonResult:
    total_questions = 0
    matching_questions = []
    differing_questions = []
    
    answers_a = {a.question_number: a for a in response_a.answers}
    answers_b = {a.question_number: a for a in response_b.answers}
    
    all_question_numbers = set(answers_a.keys()) | set(answers_b.keys())
    
    for q_num in sorted(all_question_numbers):
        total_questions += 1
        answer_a = answers_a.get(q_num)
        answer_b = answers_b.get(q_num)
        
        if answer_a and answer_b:
            match_type = _check_match(answer_a.answer, answer_b.answer)
            if match_type:
                matching_questions.append(q_num)
            else:
                differing_questions.append({
                    "question_number": q_num,
                    "question_text": answer_a.question_text,
                    "answer_a": {
                        "answer": answer_a.answer,
                        "reasoning": answer_a.reasoning,
                        "confidence": answer_a.confidence
                    },
                    "answer_b": {
                        "answer": answer_b.answer,
                        "reasoning": answer_b.reasoning,
                        "confidence": answer_b.confidence
                    }
                })
        else:
            differing_questions.append({
                "question_number": q_num,
                "question_text": answer_a.question_text if answer_a else answer_b.question_text,
                "answer_a": answer_a.model_dump() if answer_a else None,
                "answer_b": answer_b.model_dump() if answer_b else None
            })
    
    agreement = (len(matching_questions) / total_questions * 100) if total_questions > 0 else 0
    
    return ComparisonResult(
        agreement_percentage=agreement,
        matching_questions=matching_questions,
        differing_questions=differing_questions
    )

def _check_match(answer_a: str, answer_b: str) -> Optional[str]:
    normalized_a = _normalize_answer(answer_a)
    normalized_b = _normalize_answer(answer_b)
    
    if normalized_a == normalized_b:
        return "exact"
    
    if _is_numeric_match(normalized_a, normalized_b):
        return "numerical"
    
    similarity = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    if similarity >= settings.similarity_threshold:
        return "semantic"
    
    return None

def _normalize_answer(answer: str) -> str:
    # Decompose combining characters (e.g., p̂ → p + ^) then strip combining marks
    answer = unicodedata.normalize('NFKD', answer)
    answer = ''.join(c for c in answer if not unicodedata.combining(c))
    # Map subscript/superscript/Greek characters to ASCII equivalents
    answer = answer.translate(_UNICODE_TO_ASCII)
    answer = answer.strip().lower()
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'[^\w\s\-\+\=\(\)\.,]', '', answer)
    return answer

def _is_numeric_match(answer_a: str, answer_b: str) -> bool:
    numbers_a = re.findall(r'-?\d+\.?\d*', answer_a)
    numbers_b = re.findall(r'-?\d+\.?\d*', answer_b)
    
    if not numbers_a or not numbers_b:
        return False
    
    num_a = float(numbers_a[-1])
    num_b = float(numbers_b[-1])
    
    if num_a == 0 and num_b == 0:
        return True
    
    relative_diff = abs(num_a - num_b) / max(abs(num_a), abs(num_b)) if max(abs(num_a), abs(num_b)) > 0 else 0
    
    return relative_diff <= settings.numerical_tolerance

def get_disagreement_summary(differing_questions: list,
                             name_a: str = "Model A",
                             name_b: str = "Model B") -> str:
    if not differing_questions:
        return "All questions in agreement."

    summary_parts = []
    for diff in differing_questions:
        q_num = diff["question_number"]
        q_text = diff["question_text"]

        answer_a_obj = diff.get("answer_a")
        answer_a = answer_a_obj.get("answer", "N/A") if answer_a_obj else "N/A"

        answer_b_obj = diff.get("answer_b")
        answer_b = answer_b_obj.get("answer", "N/A") if answer_b_obj else "N/A"

        summary_parts.append(
            f"Question {q_num}: '{q_text}' - {name_a} says '{answer_a}', {name_b} says '{answer_b}'"
        )

    return "\n".join(summary_parts)
