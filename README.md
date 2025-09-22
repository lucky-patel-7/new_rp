# Resume Parser API

A comprehensive, production-ready resume parsing system built with FastAPI and Azure OpenAI integration.

## 🚀 Features

- **Hybrid Extraction**: Combines LLM intelligence with regex patterns for maximum accuracy
- **Multiple File Formats**: Supports PDF, DOC, DOCX, and TXT files
- **Semantic Search**: Vector-based resume search using Qdrant
- **Comprehensive Data**: Extracts work history, projects, education, skills, and more
- **Production Ready**: Proper error handling, logging, and monitoring
- **Well Organized**: Clean, maintainable codebase structure

## 📁 Project Structure

```
resume-parser/
├── src/resume_parser/          # Main package
│   ├── core/                   # Core functionality
│   ├── extractors/             # Data extraction modules
│   ├── parsers/                # File parsers
│   ├── clients/                # External service clients
│   ├── database/               # Database clients
│   └── utils/                  # Utilities
├── config/                     # Configuration
├── tests/                      # Test suite
├── docs/                       # Documentation
├── app.py                      # FastAPI application
└── requirements.txt            # Dependencies
```

## 🛠️ Quick Start

1. **Setup Environment**
   ```bash
   make setup-dev
   # Edit .env with your configuration
   ```

2. **Install Dependencies**
   ```bash
   make install
   ```

3. **Run Application**
   ```bash
   make run
   ```

4. **Check Health**
   ```bash
   make health
   ```

## 📚 Documentation

For detailed documentation, see [docs/README.md](docs/README.md)

## 🧪 Testing

```bash
make test           # Run tests
make test-coverage  # Run with coverage
make lint          # Run linting
make format        # Format code
```

## ⚙️ Configuration

Required environment variables:
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint
- `AZURE_OPENAI_CHAT_DEPLOYMENT`: Chat model deployment name
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Embedding model deployment name

See `.env.example` for complete configuration options.

## 📞 Support

- Check [docs/](docs/) for detailed documentation
- Review logs in `logs/` directory for troubleshooting
- Create issues for bugs or feature requests