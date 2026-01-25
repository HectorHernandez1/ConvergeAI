import pytest
from models import Answer, SolverResponse
from utils.comparator import compare_answers, _check_match, _normalize_answer, _is_numeric_match

@pytest.fixture
def sample_answer():
    return Answer(
        question_number=1,
        question_text="What is 2+2?",
        answer="4",
        reasoning="Basic addition",
        confidence="high"
    )

@pytest.fixture
def sample_responses():
    response_a = SolverResponse(
        model_name="OpenAI GPT-4",
        iteration=1,
        answers=[
            Answer(
                question_number=1,
                question_text="What is 2+2?",
                answer="4",
                reasoning="Basic addition",
                confidence="high"
            ),
            Answer(
                question_number=2,
                question_text="What is 3+3?",
                answer="6",
                reasoning="Basic addition",
                confidence="high"
            )
        ],
        tokens_used=100,
        cost_usd=0.001
    )
    
    response_b = SolverResponse(
        model_name="Anthropic Claude",
        iteration=1,
        answers=[
            Answer(
                question_number=1,
                question_text="What is 2+2?",
                answer="4",
                reasoning="Basic addition",
                confidence="high"
            ),
            Answer(
                question_number=2,
                question_text="What is 3+3?",
                answer="seven",
                reasoning="Spelled out number",
                confidence="medium"
            )
        ],
        tokens_used=120,
        cost_usd=0.0012
    )
    
    return response_a, response_b

def test_compare_answers_partial_agreement(sample_responses):
    response_a, response_b = sample_responses
    result = compare_answers(response_a, response_b)
    
    assert result.agreement_percentage == 50.0
    assert 1 in result.matching_questions
    assert 2 in [d["question_number"] for d in result.differing_questions]

def test_check_match_exact():
    assert _check_match("The answer is 42", "the answer is 42") == "exact"
    assert _check_match("42", "42") == "exact"

def test_check_match_numerical():
    assert _check_match("The result is 42.5", "Answer: 42.50") == "numerical"
    assert _check_match("Result: 100", "100.0") == "numerical"

def test_check_match_semantic():
    long_text_a = "The capital of France is Paris, known for the Eiffel Tower"
    long_text_b = "Paris is the capital of France, famous for the Eiffel Tower"
    assert _check_match(long_text_a, long_text_b) == "semantic"

def test_check_match_no_match():
    assert _check_match("Apple", "Banana") is None
    assert _check_match("42", "100") is None

def test_normalize_answer():
    assert _normalize_answer("  HELLO  WORLD  ") == "hello world"
    assert _normalize_answer("Answer=42!") == "answer=42"
    assert _normalize_answer("Test\n\nAnswer") == "test answer"

def test_is_numeric_match():
    assert _is_numeric_match("42", "42") == True
    assert _is_numeric_match("42.5", "42.500") == True
    assert _is_numeric_match("100", "99.9") == False
    assert _is_numeric_match("0", "0.001") == True
