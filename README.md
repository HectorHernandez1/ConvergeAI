# ConvergeAI

AI consensus problem solver that runs two models in parallel to solve academic problems through iterative consensus-building. Default pair: OpenAI GPT-5.2 and Anthropic Claude Sonnet 4.5. Also supports local open-source models via [Ollama](https://ollama.com) (Gemma, Qwen, DeepSeek-R1, Llama, etc.) — mix cloud and local, or run fully offline.

## Features

- **Pluggable Model Pair**: Any two of OpenAI, Anthropic, or local Ollama models
- **Local Open-Source Models**: Run Gemma 4, Qwen 3/3.5, DeepSeek-R1, etc. via Ollama (zero API cost)
- **Iterative Consensus**: Models compare and refine answers (up to 5 iterations) until agreement
- **Early Stop Optimization**: Stops at 90% agreement by default for cost efficiency
- **PDF Extraction**: Automatic fallback chain (PyPDF2 → pdfplumber → pymupdf)
- **PPT Support**: Extracts text from PPT/PPTX files including tables using python-pptx
- **HTML Support**: Extracts text from HTML files, removing scripts and styles
- **Vision-aware**: Auto-detects vision capability; passes images to multimodal models, skips for text-only
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

> Only needed if you plan to use OpenAI or Anthropic. If you run two local Ollama models, no API keys are required.

5. (Optional) Install [Ollama](https://ollama.com) and pull models for local inference:
```bash
# Install from https://ollama.com/download, then pull the recommended local pair:
ollama pull gemma4:31b
ollama pull qwen3.6:35b
ollama list        # confirm models are installed
```
Both models are vision-capable and ~30-35B parameters — a balanced pair for academic quizzes with figures. See the **Running with local models** section below for the override flag.

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

### Choosing the Solver Pair

**Default**: `openai` ↔ `anthropic` (cloud, requires API keys). The project was built around this pairing and that's the supported, recommended flow.

Use `--solvers` to override with any other pair. Each entry is one of:

- `openai` — uses `settings.openai_model`
- `anthropic` — uses `settings.anthropic_model`
- `ollama:<model>` — any model visible in `ollama list` (names may contain colons, e.g. `gemma4:31b`)

```bash
# Default — cloud pair (no flag needed)
python main.py --problem input/quiz.pdf

# Same thing, explicit
python main.py --solvers openai,anthropic

# Mix cloud + local — local model gets a "second opinion" from a cloud model
python main.py --solvers openai,ollama:gemma4:31b
```

### Running with local models (Ollama override)

If you'd rather skip the cloud bill (or run fully offline), override `--solvers` with two local Ollama specs. The recommended local pair is `gemma4:31b ↔ qwen3.6:35b`:

```bash
python main.py --solvers ollama:gemma4:31b,ollama:qwen3.6:35b
```

Why this pair:
- Both vision-capable (auto-detected via `ollama show`)
- Both 30–35B parameters — balanced consensus, similar inference speed
- Both fit comfortably in a 32GB-VRAM GPU; KV cache for any spilled layers goes to system RAM
- Empirically converges in 2–4 iterations on a 38-image quiz, no oscillation, **$0 cost**

Other Ollama specs you can try via the same flag:
- `ollama:qwen2.5vl:32b` — pure vision-language model, very stable, 21GB
- `ollama:llama3.2-vision:11b` — smaller, faster sanity check
- `ollama:gpt-oss:20b` — OpenAI's open-weight model

**Vision handling**: images are automatically sent only to models that declare the `vision` capability via `ollama show` (and silently skipped otherwise). If your problem has visuals, pair at least one vision model.

**Performance**: large local models (20B+) can take minutes per iteration. Default request timeout is 30 min, configurable via `ollama_timeout` in `config.py`.

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
- `SOLVERS`: Default `["openai", "anthropic"]` — the two solvers the consensus loop instantiates. Overridable via `--solvers`.
- `OLLAMA_BASE_URL`: Default `http://localhost:11434` (Ollama server URL)
- `OLLAMA_TIMEOUT`: Default `1800.0` seconds (30 min) — large local models can be slow
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
│   ├── __init__.py            # Solver factory (build_solver)
│   ├── base_solver.py         # Abstract base class
│   ├── openai_solver.py       # GPT-5 implementation
│   ├── anthropic_solver.py    # Claude implementation
│   └── ollama_solver.py       # Local Ollama models implementation
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
- Additional cloud providers (Gemini, xAI)
- Web interface for non-CLI users
- Batch processing multiple problems
- Fine-tuned prompts for specific subjects (math, coding, essays)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
