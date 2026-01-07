# Code Style & Conventions

## Python Style
- **Line length**: 88 characters (Black/Ruff)
- **Target version**: Python 3.11
- **Imports**: Absolute from `src.` (e.g., `from src.agents import ...`)
- **Type hints**: Full type annotations required (MyPy strict mode)

## Naming Conventions

### Agent Naming
| Component | Convention | Example |
|-----------|------------|---------|
| Agent Role | `{domain}-{function}-agent` | `db-troubleshoot-agent` |
| Agent Class | `{Domain}{Function}Agent` | `DbTroubleshootAgent` |
| Agent Module | `src/agents/{domain}/{function}.py` | `agents/database/troubleshoot.py` |

### Tool Naming
| Component | Convention | Example |
|-----------|------------|---------|
| MCP Tool | `oci_{domain}_{action}` | `oci_compute_list_instances` |
| Skill | `{domain}_{workflow}` | `database_rca_workflow` |

## Docstrings
- Google-style docstrings

## Error Handling
- Specific exceptions, never bare `except:`
- Avoid security vulnerabilities (OWASP top 10)

## Logging
- Use `structlog` via `get_logger()`

## Testing
- Tests mirror source structure in `tests/`
- 80%+ coverage target
