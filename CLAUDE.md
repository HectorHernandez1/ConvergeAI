# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConvergeAI is an AI consensus problem solver that uses OpenAI GPT-5.2 and Anthropic Claude Sonnet 4.5 in parallel to solve academic problems through iterative consensus-building. The system runs up to 5 iterations where both models independently solve a problem, compare their answers, and refine their responses based on disagreements until reaching consensus or hitting iteration/cost limits.

## Key Commands

### Running the Application
```bash
# Basic run (uses first PDF in input/ directory)
python main.py

# Specify problem file
python main.py --problem input/quiz.pdf

# With reference materials (auto-detected from references/ folder)
python main.py --problem input/quiz.pdf

# Custom iteration limit
python main.py --problem input/quiz.pdf --max-iterations 3

# Verbose output
python main.py --problem input/quiz.pdf --verbose

# Cache management
python main.py --problem input/quiz.pdf --clear-cache  # Clear cache before running
python main.py --problem input/quiz.pdf --no-cache     # Disable cache for this run
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
4. **Termination**: Loop exits on ≥90% agreement (configurable), 100% agreement, max iterations, or cost limit exceeded

### Solver Pattern
All AI integrations follow the `BaseSolver` abstract class:
- `solve()`: Main async method that builds prompts, calls API, parses response
- `count_tokens()`: Model-specific tokenization
- `estimate_cost()`: Per-model pricing calculation

Both `OpenAISolver` and `AnthropicSolver` implement this pattern with:
- Automatic retry logic (3 attempts with exponential backoff via tenacity)
- Response caching (file-based, 24hr TTL, disabled by default)
- Temperature adjustment (0.3 for iteration 1, 0.5 for subsequent)
- Robust JSON parsing with fallback for markdown-fenced responses and list data
- **OpenAI-specific**: Smart API parameter detection (`max_completion_tokens` for GPT-5.x models, `max_tokens` for GPT-4.x models)
- **Anthropic-specific**: Uses `max_tokens` parameter for all Claude models

### Answer Comparison Logic (utils/comparator.py)
Three matching strategies applied in order:
1. **Exact**: Normalized string equality (lowercase, whitespace-collapsed)
2. **Numerical**: Extracts numbers, checks relative difference ≤1% (configurable)
3. **Semantic**: SequenceMatcher ratio ≥85% similarity (configurable)

Disagreement summaries are structured as: "Question N: 'text' - GPT says 'X', Claude says 'Y'"

### PDF/PPT Extraction Chain (utils/pdf_reader.py)
**PDF Fallback Waterfall:**
1. PyPDF2 (fastest, basic extraction)
2. pdfplumber (better table/layout handling)
3. pymupdf (most robust, OCR-capable)

**PPT/PPTX Extraction:**
- Extracts text from all slides using python-pptx
- Handles table data via `shape.has_table` property
- Efficient file type routing based on extension
- Warns for .ppt files (limited support; recommends .pptx conversion)

### Response Caching
- Cache key: SHA256 hash of (model_name + prompt) using surrogatepass encoding for Unicode
- Storage: `logs/cache/{hash}.json`
- Invalidation: File timestamp checked against TTL (default 24hr)
- Default: **Disabled** (edit `config.py` to enable for development: `enable_cache: bool = True`)
- CLI flags: `--clear-cache` (clear before run), `--no-cache` (disable for run)
- Unicode support: Handles mathematical symbols (e.g., ∑, ∫, 𝕩) from PDF/PPT extraction

## Configuration

**`.env` file (required):** API keys only
```env
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
```

**To customize settings:** Edit `config.py` directly (do NOT add these to `.env`)

**`config.py` defaults:**
- `OPENAI_MODEL`: Default `gpt-5.2` (most capable)
  - gpt-5.2: $0.00175/1K input, $0.014/1K output
  - gpt-5.1: $0.00125/1K input, $0.01/1K output
  - gpt-4.1-nano: $0.0001/1K input, $0.0004/1K output (cost-effective)
- `ANTHROPIC_MODEL`: Default `claude-sonnet-4-5` (balanced)
  - claude-sonnet-4-5: $0.003/1K input, $0.015/1K output
  - claude-haiku-4-5: $0.001/1K input, $0.005/1K output (cost-effective)
  - claude-opus-4-5: $0.005/1K input, $0.025/1K output (most capable)
- `MAX_TOKENS`: Default `8000` (max response tokens for all models)
- `MAX_ITERATIONS`: Default `5`
- `EARLY_STOP_THRESHOLD`: Default `0.90` (90% agreement - stops iteration early for efficiency)
- `MAX_COST_USD`: Default `5.0` (hard stop to prevent runaway costs)
- `AGREEMENT_THRESHOLD`: Default `1.0` (100% agreement required for full consensus)
- `SIMILARITY_THRESHOLD`: Default `0.85` (semantic matching threshold)
- `NUMERICAL_TOLERANCE`: Default `0.01` (1% tolerance for numeric answers)
- `ENABLE_CACHE`: Default `false` (disabled in production; edit config.py to enable for development)

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

### Early Stop Threshold
The system includes an **early stop threshold** (default 90%) for production efficiency:
- Consensus loop exits when agreement reaches `EARLY_STOP_THRESHOLD` (not just 100%)
- Prevents unnecessary iterations when models are "close enough"
- Configurable in `config.py`: set `early_stop_threshold: float = 1.0` for strict 100% requirement
- Reduces API costs while maintaining quality (e.g., 9/10 questions agreed is often sufficient)

### Caching Strategy
**Default: Disabled** (`enable_cache: bool = False` in config.py)
- Production: Disabled by default to ensure fresh analysis for unique academic problems
- Development: Enable in `config.py` by setting `enable_cache: bool = True` to save costs during testing
- Each problem+references combination is typically unique, making cache hits rare
- CLI flags override config: `--no-cache` forces disable, `--clear-cache` purges before run

**When to Enable Caching:**
- Repeated testing of same problem file during development
- Re-running failed iterations with identical prompts
- Cost-sensitive applications where staleness is acceptable

### Token Counting
- Both OpenAI and Anthropic: Uses actual token counts from API responses (`response.usage`)
- OpenAI: `prompt_tokens` and `completion_tokens` from API response
- Anthropic: `input_tokens` and `output_tokens` from API response
- This ensures 100% accurate cost tracking without estimation

### Prompt Templates
Two distinct prompts in `prompts/`:
- `initial_solve.txt`: First iteration with problem + references
- `refinement.txt`: Subsequent iterations with previous answers + disagreement summary

Variables are injected via `.format()` (not f-strings) to allow file-based templates.

### Error Handling
- **API errors**: 3 retries with exponential backoff (1s to 10s)
- **JSON parsing**: Multi-stage fallback (raw JSON → markdown fence extraction → brace-matching → error with preview)
- **List data handling**: Automatically wraps list responses in `{"answers": [...]}` structure
- **Unicode encoding**: Surrogatepass encoding handles mathematical symbols and special characters from PDFs
- **PDF/PPT extraction**: Silent fallback through extraction chain with file type routing
- **Model API compatibility**: Automatic parameter selection for GPT-5.x vs GPT-4.x models
- **Token limits**: Configurable max_tokens (default 8000) to handle lengthy responses with references
- **Cost overruns**: Graceful early exit with partial results
