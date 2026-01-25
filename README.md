# ConvergeAI

AI consensus problem solver using OpenAI GPT-5.2 and Anthropic Claude Sonnet 4.5 to solve academic problems through iterative consensus-building.

## Features

- **Dual Model Approach**: Leverages both OpenAI GPT and Anthropic Claude for robust solutions
- **Iterative Consensus**: Models compare and refine answers (up to 5 iterations) until agreement
- **Early Stop Optimization**: Stops at 90% agreement by default for cost efficiency
- **PDF Extraction**: Automatic fallback chain (PyPDF2 → pdfplumber → pymupdf)
- **PPT Support**: Extracts text from PPT/PPTX files including tables using python-pptx
- **HTML Support**: Extracts text from HTML files, removing scripts and styles
- **Optional Caching**: Reduces API costs for repeat queries (disabled by default in production)
- **Cache Management**: CLI flags for clearing cache or disabling per-run
- **Token & Cost Tracking**: Real-time monitoring of API usage and costs
- **Rich CLI Output**: Beautiful terminal interface with progress tracking
- **JSON + Markdown Export**: Results saved in both formats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ConvergeAI.git
cd ConvergeAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key
```

## Usage

### Basic Usage

```bash
# PDF file
python main.py --problem input/quiz.pdf

# PPT/PPTX file (also supported)
python main.py --problem input/lecture.pptx

# HTML file (also supported)
python main.py --problem input/webpage.html
```

### Supported File Types

ConvergeAI supports the following document types:
- **PDF files** (.pdf): Uses automatic fallback extraction chain
- **PowerPoint files** (.ppt, .pptx): Extracts text from slides and tables
- **HTML files** (.html, .htm): Extracts text, removes scripts and styles
- **Reference materials**: Optional - place PDF/PPT/HTML files in `references/` folder

### With Reference Materials (Optional)

ConvergeAI automatically detects and uses all PDFs, PPTs, or HTML files in the `references/` folder:

```bash
python main.py --problem input/quiz.pdf
```

Add PDFs to the `references/` folder and ConvergeAI will automatically extract and combine them for the AI models.

**Note:** Reference materials are optional. If no PDFs are found in `references/`, the system runs without references.

### Custom Settings

```bash
python main.py --problem input/quiz.py \
    --max-iterations 3 \
    --max-cost 2.0
```

### Verbose Mode

```bash
python main.py --problem input/quiz.pdf --verbose
```

### Cache Management

```bash
# Clear cache before running
python main.py --problem input/quiz.pdf --clear-cache

# Disable cache for this run
python main.py --problem input/quiz.pdf --no-cache
```

## Configuration

### About config.py

The [config.py](config.py) file uses Pydantic Settings to manage all configuration with sensible production defaults.

**Key features:**
- Type-safe configuration with Pydantic validation
- Production-ready defaults (balanced models, caching disabled)
- Edit [config.py](config.py) directly to customize settings

### API Keys (.env file)

The `.env` file is **only for API keys**. Edit `.env` to add your credentials:

```env
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key
```

### Application Settings (config.py)

To change application settings, edit [config.py](config.py) directly. Available settings and their defaults:

- `OPENAI_MODEL`: Default `gpt-5.2` (also available: `gpt-5.1`, `gpt-4.1-nano`)
  - gpt-5.2: $0.00175/1K input, $0.014/1K output (most capable)
  - gpt-5.1: $0.00125/1K input, $0.01/1K output
  - gpt-4.1-nano: $0.0001/1K input, $0.0004/1K output (cost-effective)
- `ANTHROPIC_MODEL`: Default `claude-sonnet-4-5` (also available: `claude-haiku-4-5`, `claude-opus-4-5`)
  - claude-sonnet-4-5: $0.003/1K input, $0.015/1K output (balanced default)
  - claude-haiku-4-5: $0.001/1K input, $0.005/1K output (cost-effective)
  - claude-opus-4-5: $0.005/1K input, $0.025/1K output (most capable)
- `MAX_TOKENS`: Default `8000` (max response tokens for all models)
- `MAX_ITERATIONS`: Default `5`
- `EARLY_STOP_THRESHOLD`: Default `0.90` (90% - stops iteration early for efficiency)
- `MAX_COST_USD`: Default `5.0`
- `AGREEMENT_THRESHOLD`: Default `1.0` (100%)
- `ENABLE_CACHE`: Default `false` (disabled for production; edit config.py to enable for development)
- `SIMILARITY_THRESHOLD`: Default `0.85` (semantic matching threshold)
- `NUMERICAL_TOLERANCE`: Default `0.01` (1% tolerance for numeric answers)

**To customize settings:** Edit the values directly in [config.py](config.py). For example, to use cost-effective models for testing:
```python
openai_model: str = "gpt-4.1-nano"
anthropic_model: str = "claude-haiku-4-5"
enable_cache: bool = True
```

## Output

Results are saved in the `output/` directory:

- `*_solutions_TIMESTAMP.json`: Complete data including iteration history
- `*_solutions_TIMESTAMP.md`: Human-readable report

### Sample Output

```json
{
  "timestamp": "2025-01-24T15:30:00Z",
  "iterations_needed": 3,
  "final_agreement": 100.0,
  "total_cost_usd": 0.47,
  "consensus_answers": [...],
  "model_responses": {...}
}
```

## Project Structure

```
ConvergeAI/
├── main.py                    # CLI entry point and orchestration
├── config.py                  # Configuration and environment loading
├── models.py                  # Pydantic models for type safety
├── solvers/
│   ├── base_solver.py         # Abstract base class
│   ├── openai_solver.py       # GPT-5 implementation
│   └── anthropic_solver.py    # Claude implementation
 ├── utils/
 │   ├── file_reader.py         # File extraction with fallback chain (PDF/PPT/HTML)
 │   ├── comparator.py          # Answer comparison logic
 │   ├── cache.py               # Response caching
 │   └── token_counter.py       # Token usage tracking
 ├── prompts/
 │   ├── initial_solve.txt      # First iteration prompt
 │   └── refinement.txt         # Subsequent iteration prompt
 ├── input/                     # Place problem PDFs here (NOT reference materials)
 ├── references/                 # Place reference materials PDFs/PPTs here (optional, auto-detected)
 ├── output/                    # Results saved here
 ├── logs/                      # Iteration logs and cache
├── tests/                     # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## Testing

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=. --cov-report=html
```

## Error Handling

- **File extraction failures**: Clear error message with fallback chain (PDF/PPT/HTML)
- **Unicode encoding**: Handles mathematical symbols and special characters with surrogatepass encoding
- **API rate limits**: Exponential backoff with retries (max 3)
- **Invalid JSON**: Extracts from markdown fences, handles list data, shows preview on failure
- **Model compatibility**: Smart API parameter detection (max_completion_tokens for GPT-5.x, max_tokens for GPT-4.x)
- **Cost limit exceeded**: Stop iteration, return best result
- **Network errors**: Retry with backoff, timeout after 60s

## Future Enhancements

- Support for image-based PDFs (OCR integration)
- Additional models (Gemini, Llama via API)
- Web interface for non-CLI users
- Batch processing multiple problems
- Fine-tuned prompts for specific subjects (math, coding, essays)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
