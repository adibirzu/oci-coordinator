# Development Commands

## Setup
```bash
poetry install                    # Install dependencies
```

## Running
```bash
poetry run python -m src.main                # Start coordinator (Slack + API on port 3001)
poetry run python -m src.main --mode both    # Slack + API (default)
poetry run python -m src.main --mode slack   # Slack only
poetry run python -m src.main --mode api     # API only
poetry run python -m src.main --port 8080    # Custom API port
```

## Testing
```bash
poetry run pytest                             # Run all tests
poetry run pytest --cov=src                   # With coverage
poetry run pytest tests/test_mcp_server.py -v # Specific test
poetry run pytest --cov=src --cov-report=html # HTML coverage report
```

## Linting & Formatting
```bash
poetry run ruff check src tests        # Lint with ruff
poetry run ruff check --fix src tests  # Auto-fix ruff issues
poetry run black src tests             # Format with black
poetry run mypy src                    # Type checking
```

## OCA Authentication
```bash
poetry run python scripts/oca_auth.py           # Full OAuth login
poetry run python scripts/oca_auth.py --status  # Check token status
```

## Utility (Darwin/macOS)
```bash
git status           # Git status
git diff             # Show changes
ls -la               # List files
find . -name "*.py"  # Find Python files
grep -r "pattern" .  # Search in files
```
