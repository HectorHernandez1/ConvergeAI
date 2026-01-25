# ConvergeAI

AI consensus problem solver using OpenAI GPT-5.2 and Anthropic Claude Sonnet 4.5 to solve academic problems through iterative consensus-building.

## Features

- **Dual Model Approach**: Leverages both OpenAI GPT-5.2 and Anthropic Claude Sonnet 4.5 for robust solutions
- **Iterative Consensus**: Models compare and refine answers (up to 5 iterations) until agreement
- **PDF Extraction**: Automatic fallback chain (PyPDF2 → pdfplumber → pymupdf)
- **PPT Support**: Extracts text from PPT/PPTX files using python-pptx
- **Response Caching**: Reduces API costs for repeat queries
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
```

### Supported File Types

ConvergeAI supports the following document types:
- **PDF files** (.pdf): Uses automatic fallback extraction chain
- **PowerPoint files** (.ppt, .pptx): Extracts text from slides and tables
- **Reference materials**: Optional - place PDF/PPT files in `references/` folder

### With Reference Materials (Optional)

ConvergeAI automatically detects and uses all PDFs or PPT files in the `references/` folder:

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

## Configuration

Edit `.env` file to configure:

- `OPENAI_MODEL`: Default `gpt-5.2` (also available: `gpt-5.1`, `gpt-4.1-nano`)
- `ANTHROPIC_MODEL`: Default `claude-sonnet-4-5` (also available: `claude-haiku-4-5`, `claude-opus-4-5`)
- `MAX_ITERATIONS`: Default `5`
- `MAX_COST_USD`: Default `5.0`
- `AGREEMENT_THRESHOLD`: Default `1.0` (100%)
- `ENABLE_CACHE`: Default `true`

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
│   ├── pdf_reader.py          # PDF extraction with fallback chain
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

- **PDF extraction failures**: Clear error message with suggestions
- **API rate limits**: Exponential backoff with retries (max 3)
- **Invalid JSON**: Retry with stricter prompt, then fail gracefully
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
