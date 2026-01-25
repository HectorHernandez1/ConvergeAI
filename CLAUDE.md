# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConvergeAI is an AI consensus problem solver that uses OpenAI GPT-4 and Anthropic Claude in parallel to solve academic problems through iterative consensus-building. The system runs up to 5 iterations where both models independently solve a problem, compare their answers, and refine their responses based on disagreements until reaching consensus or hitting iteration/cost limits.

## Key Commands

### Running the Application
```bash
# Basic run (uses first PDF in input/ directory)
python main.py

# Specify problem file
python main.py --problem input/quiz.pdf

# With reference materials
python main.py --problem input/quiz.pdf --references input/textbook.pdf

# Custom iteration limit
python main.py --problem input/quiz.pdf --max-iterations 3

# Verbose output
python main.py --problem input/quiz.pdf --verbose
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_solvers.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run async tests
pytest tests/test_solvers.py -v
```

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Then edit .env with API keys
```

## Architecture

### Core Consensus Loop (main.py)
The `run_consensus()` function orchestrates the entire process:
1. **Iteration 1**: Both models solve independently using `prompts/initial_solve.txt`
2. **Iterations 2+**: Models receive their previous answer, the other model's answer, and a disagreement summary via `prompts/refinement.txt`
3. **Comparison**: After each iteration, `comparator.py` checks answer agreement (exact, numerical, semantic matching)
4. **Termination**: Loop exits on 100% agreement, max iterations, or cost limit exceeded

### Solver Pattern
All AI integrations follow the `BaseSolver` abstract class:
- `solve()`: Main async method that builds prompts, calls API, parses response
- `count_tokens()`: Model-specific tokenization
- `estimate_cost()`: Per-model pricing calculation

Both `OpenAISolver` and `AnthropicSolver` implement this pattern with:
- Automatic retry logic (3 attempts with exponential backoff via tenacity)
- Response caching (file-based, 24hr TTL)
- Temperature adjustment (0.3 for iteration 1, 0.5 for subsequent)
- Robust JSON parsing with fallback for markdown-fenced responses

### Answer Comparison Logic (utils/comparator.py)
Three matching strategies applied in order:
1. **Exact**: Normalized string equality (lowercase, whitespace-collapsed)
2. **Numerical**: Extracts numbers, checks relative difference ≤1% (configurable)
3. **Semantic**: SequenceMatcher ratio ≥85% similarity (configurable)

Disagreement summaries are structured as: "Question N: 'text' - GPT says 'X', Claude says 'Y'"

### PDF Extraction Chain (utils/pdf_reader.py)
Fallback waterfall for robust extraction:
1. PyPDF2 (fastest, basic extraction)
2. pdfplumber (better table/layout handling)
3. pymupdf (most robust, OCR-capable)

### Response Caching
- Cache key: SHA256 hash of (model_name + prompt)
- Storage: `logs/cache/{hash}.json`
- Invalidation: File timestamp checked against TTL (default 24hr)
- Controlled via `ENABLE_CACHE` environment variable

## Configuration (.env)

Critical settings in `config.py` (loaded from `.env`):
- `OPENAI_MODEL`: Default `gpt-4o`
 - `ANTHROPIC_MODEL`: Default `claude-sonnet-4-5`
- `MAX_ITERATIONS`: Default `5`
- `MAX_COST_USD`: Default `5.0` (hard stop to prevent runaway costs)
- `AGREEMENT_THRESHOLD`: Default `1.0` (100% agreement required)
- `SIMILARITY_THRESHOLD`: Default `0.85` (semantic matching threshold)
- `NUMERICAL_TOLERANCE`: Default `0.01` (1% tolerance for numeric answers)

## Data Models (models.py)

All data structures use Pydantic for validation:
- `Answer`: Single question-answer pair with reasoning, confidence, references
- `SolverResponse`: Complete model output for one iteration
- `ComparisonResult`: Agreement metrics and differing question details
- `FinalOutput`: Top-level result with consensus answers, iteration history, costs
- `IterationLog`: Snapshot of one consensus iteration

## Output Structure

Two formats generated in `output/`:
1. **JSON** (`*_solutions_TIMESTAMP.json`): Complete data including all iterations, token usage, costs per model
2. **Markdown** (`*_solutions_TIMESTAMP.md`): Human-readable report with consensus answers and iteration history

## Testing Patterns

Tests use `pytest-asyncio` for async solver methods. Key test files:
- `tests/test_solvers.py`: Solver implementations and API interactions
- `tests/test_comparator.py`: Answer matching logic (exact, numerical, semantic)
- `tests/test_pdf_reader.py`: PDF extraction fallback chain

## Important Implementation Notes

### Consensus Determination (main.py:_determine_consensus)
When agreement isn't 100%, final answers are chosen by:
1. Use matching answers for agreed questions
2. For disagreements: prefer the answer with "high" confidence
3. If both have same confidence level: default to OpenAI's answer

### Token Counting
- OpenAI: Uses `tiktoken` library via `utils/token_counter.py`
- Anthropic: Uses `anthropic.Anthropic().count_tokens()`

### Prompt Templates
Two distinct prompts in `prompts/`:
- `initial_solve.txt`: First iteration with problem + references
- `refinement.txt`: Subsequent iterations with previous answers + disagreement summary

Variables are injected via `.format()` (not f-strings) to allow file-based templates.

### Error Handling
- API errors: 3 retries with exponential backoff (1s to 10s)
- JSON parsing: Attempts to extract from markdown fences if raw parse fails
- PDF extraction: Silent fallback through extraction chain
- Cost overruns: Graceful early exit with partial results
