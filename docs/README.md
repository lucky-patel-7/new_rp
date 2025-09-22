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
│   │   ├── models.py           # Data models
│   │   └── parser.py           # Main parser class
│   ├── extractors/             # Data extraction modules
│   │   └── hybrid_extractor.py # Hybrid LLM+regex extractor
│   ├── parsers/                # File parsers
│   │   └── section_parser.py   # Section-based parsing
│   ├── clients/                # External service clients
│   │   └── azure_openai.py     # Azure OpenAI client
│   ├── database/               # Database clients
│   │   └── qdrant_client.py    # Qdrant vector database
│   └── utils/                  # Utilities
│       ├── logging.py          # Logging configuration
│       └── file_handler.py     # File processing
├── config/                     # Configuration
│   └── settings.py             # Application settings
├── tests/                      # Test suite
├── docs/                       # Documentation
├── logs/                       # Log files
├── app.py                      # FastAPI application
├── .env                        # Environment variables
└── requirements.txt            # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd resume-parser
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## ⚙️ Configuration

### Required Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=resumes

# PostgreSQL (optional)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=resume_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# Application
APP_NAME=Resume Parser API
DEBUG=false
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
```

## 🚀 Running the Application

### Development
```bash
python app.py
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📚 API Endpoints

### Upload Resume
```http
POST /upload-resume
Content-Type: multipart/form-data

file: resume.pdf
```

### Search Resumes
```http
POST /search-resumes
Content-Type: application/json

{
    "query": "Python developer with machine learning experience",
    "limit": 10,
    "role_filter": "Software Engineer",
    "seniority_filter": "Senior"
}
```

### Get Resume
```http
GET /resume/{user_id}
```

### Health Check
```http
GET /health
```

## 🎯 Usage Examples

### Upload a Resume
```python
import requests

with open("resume.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-resume",
        files={"file": f}
    )

result = response.json()
print(f"User ID: {result['user_id']}")
```

### Search for Similar Resumes
```python
response = requests.post(
    "http://localhost:8000/search-resumes",
    json={
        "query": "Senior Python developer with 5+ years experience",
        "limit": 5
    }
)

results = response.json()
for resume in results["results"]:
    print(f"Score: {resume['score']}, Name: {resume['payload']['name']}")
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src/resume_parser
```

## 📊 Monitoring and Logging

- Logs are written to `logs/` directory
- Application logs: `logs/resume_parser.log`
- Error logs: `logs/errors.log`
- Health check endpoint: `/health`

## 🔧 Development

### Code Style
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch
2. Add implementation in appropriate module
3. Add tests
4. Update documentation
5. Submit pull request

## 🐛 Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Issues**
   - Verify API key and endpoint
   - Check deployment names
   - Ensure network connectivity

2. **Qdrant Connection Issues**
   - Verify Qdrant is running
   - Check host/port configuration
   - Ensure collection exists

3. **File Upload Issues**
   - Check file size limits
   - Verify file type support
   - Check disk space

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the logs for error details