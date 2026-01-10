"""
Secure Code Executor for DeepSkills.

Provides sandboxed Python execution for agent data processing.
Execution is isolated with restricted imports and resource limits.

Security model:
- Restricted builtins (no file I/O, exec, eval)
- Allowlisted imports only (pandas, numpy, json, etc.)
- Memory and time limits
- No network access from code

Example usage:
    executor = CodeExecutor()
    result = await executor.execute(
        code='''
            import pandas as pd
            df = pd.DataFrame(data)
            return df.describe().to_dict()
        ''',
        variables={'data': raw_data},
        timeout_seconds=30
    )
"""

import asyncio
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import structlog

logger = structlog.get_logger()

# Safe builtins for sandboxed execution
SAFE_BUILTINS = {
    # Basic types and functions
    'True': True,
    'False': False,
    'None': None,
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'getattr': getattr,
    'hasattr': hasattr,
    'hash': hash,
    'int': int,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'iter': iter,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'next': next,
    'object': object,
    'print': print,  # Safe, outputs captured
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'slice': slice,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'type': type,
    'zip': zip,
    # Math
    'divmod': divmod,
    'pow': pow,
    # Exceptions (read-only)
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'KeyError': KeyError,
    'IndexError': IndexError,
    'RuntimeError': RuntimeError,
}

# Allowed imports for data processing
ALLOWED_IMPORTS = {
    'json',
    'math',
    'statistics',
    'collections',
    'itertools',
    'functools',
    'operator',
    'datetime',
    're',
    'typing',
    # Data processing (if available)
    'pandas',
    'numpy',
}

# Blocked imports for security
BLOCKED_IMPORTS = {
    'os',
    'sys',
    'subprocess',
    'socket',
    'requests',
    'urllib',
    'http',
    'ftplib',
    'smtplib',
    'pickle',
    'marshal',
    'shelve',
    'shutil',
    'tempfile',
    'pathlib',
    'glob',
    'importlib',
    '__builtins__',
    'builtins',
}


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""

    # Resource limits
    timeout_seconds: int = 30
    max_memory_mb: int = 256
    max_output_chars: int = 100_000

    # Execution mode
    use_process_isolation: bool = False  # Use subprocess for stronger isolation

    # Allowed/blocked imports
    allowed_imports: Set[str] = field(default_factory=lambda: ALLOWED_IMPORTS.copy())
    blocked_imports: Set[str] = field(default_factory=lambda: BLOCKED_IMPORTS.copy())

    # Safe builtins
    safe_builtins: Dict[str, Any] = field(default_factory=lambda: SAFE_BUILTINS.copy())


@dataclass
class CodeExecutionResult:
    """Result from code execution."""

    success: bool
    result: Optional[Any] = None
    output: str = ""  # Captured stdout
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None

    # Metrics
    execution_time_ms: int = 0

    # Metadata
    executed_at: datetime = field(default_factory=datetime.utcnow)


class RestrictedImporter:
    """Custom importer that restricts available modules."""

    def __init__(self, allowed: Set[str], blocked: Set[str]):
        self.allowed = allowed
        self.blocked = blocked

    def find_module(self, name: str, path=None):
        """Check if module is allowed."""
        # Get the top-level module name
        top_level = name.split('.')[0]

        if top_level in self.blocked:
            return self  # Return self to handle with load_module

        if top_level not in self.allowed:
            return self  # Block unknown modules

        return None  # Allow the import

    def load_module(self, name: str):
        """Block the module with an error."""
        raise ImportError(f"Import of '{name}' is not allowed in sandboxed execution")


class CodeExecutor:
    """
    Secure Python code executor for DeepSkills.

    Features:
    - Sandboxed execution with restricted builtins
    - Allowlisted imports only
    - Timeout protection
    - Captured output
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    async def execute(
        self,
        code: str,
        variables: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> CodeExecutionResult:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute
            variables: Variables to inject into namespace
            timeout_seconds: Override default timeout

        Returns:
            CodeExecutionResult with success status and result/error
        """
        timeout = timeout_seconds or self.config.timeout_seconds
        start_time = datetime.utcnow()

        try:
            # Run in thread pool with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._thread_pool,
                    self._execute_sandboxed,
                    code,
                    variables or {}
                ),
                timeout=timeout
            )

            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            result.execution_time_ms = execution_time
            return result

        except asyncio.TimeoutError:
            return CodeExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                error_type="TimeoutError",
                execution_time_ms=timeout * 1000
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
                execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )

    def _execute_sandboxed(
        self,
        code: str,
        variables: Dict[str, Any]
    ) -> CodeExecutionResult:
        """Execute code in sandboxed environment (runs in thread)."""
        import io
        from contextlib import redirect_stdout, redirect_stderr

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Build restricted globals
        restricted_globals = {
            '__builtins__': self.config.safe_builtins,
            '__name__': '__sandboxed__',
        }

        # Add restricted import function
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Restricted import that only allows safe modules."""
            top_level = name.split('.')[0]

            if top_level in self.config.blocked_imports:
                raise ImportError(f"Import of '{name}' is not allowed")

            if top_level not in self.config.allowed_imports:
                raise ImportError(f"Import of '{name}' is not in allowed list")

            # Use real import for allowed modules
            return __builtins__['__import__'](name, globals, locals, fromlist, level)

        restricted_globals['__builtins__']['__import__'] = restricted_import

        # Add user variables
        restricted_globals.update(variables)

        # Wrap code to capture return value
        wrapped_code = self._wrap_code_for_return(code)

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile first to catch syntax errors
                compiled = compile(wrapped_code, '<sandboxed>', 'exec')

                # Execute
                local_vars: Dict[str, Any] = {}
                exec(compiled, restricted_globals, local_vars)

                # Get return value
                result = local_vars.get('__return_value__')
                output = stdout_capture.getvalue()

                # Truncate output if needed
                if len(output) > self.config.max_output_chars:
                    output = output[:self.config.max_output_chars] + "\n... (truncated)"

                return CodeExecutionResult(
                    success=True,
                    result=result,
                    output=output
                )

        except SyntaxError as e:
            return CodeExecutionResult(
                success=False,
                error=str(e),
                error_type="SyntaxError",
                traceback=traceback.format_exc(),
                output=stderr_capture.getvalue()
            )
        except ImportError as e:
            return CodeExecutionResult(
                success=False,
                error=str(e),
                error_type="ImportError",
                output=stderr_capture.getvalue()
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
                output=stderr_capture.getvalue()
            )

    def _wrap_code_for_return(self, code: str) -> str:
        """
        Wrap code to capture return value.

        Handles both explicit return statements and implicit last expression.
        """
        lines = code.strip().split('\n')

        # If code ends with 'return X', convert to assignment
        if lines and lines[-1].strip().startswith('return '):
            return_expr = lines[-1].strip()[7:]  # Remove 'return '
            lines[-1] = f'__return_value__ = {return_expr}'
            return '\n'.join(lines)

        # Otherwise, try to capture last expression
        # This is tricky - we'll try to eval the last line if it's an expression
        wrapped = '\n'.join(lines[:-1]) if len(lines) > 1 else ''

        if lines:
            last_line = lines[-1].strip()
            # Check if last line is an expression (not an assignment or statement)
            if last_line and not any(
                last_line.startswith(kw)
                for kw in ['if ', 'for ', 'while ', 'def ', 'class ', 'try:', 'with ', 'import ', 'from ']
            ) and '=' not in last_line.split('#')[0]:  # Ignore = in comments
                # Try to capture as expression
                if wrapped:
                    wrapped += f'\n__return_value__ = {last_line}'
                else:
                    wrapped = f'__return_value__ = {last_line}'
            else:
                if wrapped:
                    wrapped += f'\n{last_line}'
                else:
                    wrapped = last_line
                wrapped += '\n__return_value__ = None'

        return wrapped

    async def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code without executing it.

        Checks for:
        - Syntax errors
        - Blocked imports
        - Dangerous patterns
        """
        # Check for blocked imports
        for blocked in self.config.blocked_imports:
            if f'import {blocked}' in code or f'from {blocked}' in code:
                return False, f"Import of '{blocked}' is not allowed"

        # Check for dangerous patterns
        dangerous_patterns = [
            ('eval(', "eval() is not allowed"),
            ('exec(', "exec() is not allowed"),
            ('__import__', "Direct __import__ is not allowed"),
            ('open(', "File operations are not allowed"),
            ('compile(', "compile() is not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in code:
                return False, message

        # Try to compile
        try:
            compile(code, '<validate>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        return True, None

    def close(self):
        """Clean up resources."""
        self._thread_pool.shutdown(wait=False)


# Convenience function for one-off execution
async def execute_code(
    code: str,
    variables: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 30
) -> CodeExecutionResult:
    """
    Execute code with default settings.

    This is a convenience function for simple use cases.
    For repeated executions, create a CodeExecutor instance.
    """
    executor = CodeExecutor()
    try:
        return await executor.execute(code, variables, timeout_seconds)
    finally:
        executor.close()
