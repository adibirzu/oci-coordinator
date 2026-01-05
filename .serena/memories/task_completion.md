# Task Completion Checklist

When completing a coding task, run the following checks:

## 1. Linting & Formatting
```bash
poetry run ruff check src tests --fix  # Auto-fix lint issues
poetry run black src tests             # Format code
```

## 2. Type Checking
```bash
poetry run mypy src                    # Ensure no type errors
```

## 3. Testing
```bash
poetry run pytest --cov=src            # Run tests with coverage
```

## 4. Verify Changes
- Ensure backward compatibility or update all references
- Check for security vulnerabilities
- Verify no bare `except:` blocks
- Confirm type hints are complete

## 5. Git (if committing)
```bash
git diff                               # Review changes
git add -A && git commit -m "..."      # Commit with descriptive message
```

## Notes
- Coverage target: 80%+
- Don't create new files unless necessary
- Prefer editing existing code
- Use symbolic editing tools when possible
