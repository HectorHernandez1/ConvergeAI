import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from models import Answer, SolverResponse
from solvers.openai_solver import OpenAISolver
from solvers.anthropic_solver import AnthropicSolver

@pytest.fixture
def mock_openai_response():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"answers": [{"question_number": 1, "question_text": "Test", "answer": "42", "reasoning": "Test", "confidence": "high"}]}'
    return mock_response

@pytest.mark.asyncio
async def test_openai_solver_solve(mock_openai_response):
    with patch.object(OpenAISolver, '__init__', lambda self: None):
        solver = OpenAISolver()
        solver.model_name = "OpenAI GPT-4"
        solver.model = "gpt-4o"
        solver.client = AsyncMock()
        solver.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        solver.count_tokens = lambda x: 50
        solver.estimate_cost = lambda i, o: 0.001
        
        with patch('solvers.openai_solver.get_cached_response', return_value=None):
            with patch('solvers.openai_solver.cache_response'):
                response = await solver.solve("Test problem", iteration=1)
                
                assert response.model_name == "OpenAI GPT-4"
                assert response.iteration == 1
                assert len(response.answers) == 1
                assert response.answers[0].answer == "42"

@pytest.mark.asyncio
async def test_anthropic_solver_solve():
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = '{"answers": [{"question_number": 1, "question_text": "Test", "answer": "42", "reasoning": "Test", "confidence": "high"}]}'
    
    with patch.object(AnthropicSolver, '__init__', lambda self: None):
        solver = AnthropicSolver()
        solver.model_name = "Anthropic Claude"
        solver.model = "claude-sonnet-4-5"
        solver.client = AsyncMock()
        solver.client.messages.create = AsyncMock(return_value=mock_response)
        solver.count_tokens = lambda x: 50
        solver.estimate_cost = lambda i, o: 0.001
        
        with patch('solvers.anthropic_solver.get_cached_response', return_value=None):
            with patch('solvers.anthropic_solver.cache_response'):
                response = await solver.solve("Test problem", iteration=1)
                
                assert response.model_name == "Anthropic Claude"
                assert response.iteration == 1
                assert len(response.answers) == 1
                assert response.answers[0].answer == "42"

def test_openai_estimate_cost():
    solver = OpenAISolver()
    solver.model = "gpt-4o"
    
    cost = solver.estimate_cost(1000, 500)
    assert cost > 0

def test_anthropic_estimate_cost():
    solver = AnthropicSolver()
    solver.model = "claude-sonnet-4-5"
    
    cost = solver.estimate_cost(1000, 500)
    assert cost > 0

def test_parse_response_valid_json():
    from solvers.openai_solver import OpenAISolver
    solver = OpenAISolver()
    
    result = solver._parse_response('{"answers": []}')
    assert "answers" in result

def test_parse_response_json_with_markdown():
    from solvers.openai_solver import OpenAISolver
    solver = OpenAISolver()
    
    result = solver._parse_response('```json\n{"answers": []}\n```')
    assert "answers" in result

def test_parse_response_invalid_json():
    from solvers.openai_solver import OpenAISolver
    solver = OpenAISolver()
    
    with pytest.raises(ValueError):
        solver._parse_response('invalid json')
