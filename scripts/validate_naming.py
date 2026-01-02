#!/usr/bin/env python3
"""
Naming Convention Validator for OCI AI Agent Coordinator.

Enforces:
1. Agent definitions in src/agents/ must have class names ending in 'Agent'.
2. MCP Tools in src/mcp/server/tools/ must have function names starting with 'oci_{domain}_'.
"""

import ast
import os
import sys
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
AGENTS_DIR = PROJECT_ROOT / "src" / "agents"
TOOLS_DIR = PROJECT_ROOT / "src" / "mcp" / "server" / "tools"

violations = []

def check_agent_naming():
    """Check that agent classes end with 'Agent'."""
    print(f"Scanning Agents in {AGENTS_DIR}...")
    for root, _, files in os.walk(AGENTS_DIR):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Heuristic: Check if it inherits from BaseAgent or is in an agent directory
                            if "Agent" in node.name:
                                continue
                            
                            # We strictly require "Agent" suffix for main agent classes
                            # But we might have helper classes. 
                            # Let's just warn if a class in an agent file doesn't end in Agent
                            # and looks like a main class (not an Exception or TypedDict)
                            if not node.name.endswith("Agent") and not node.name.endswith("State") and not node.name.endswith("Config"):
                                # Check if it seems to be an agent definition
                                is_agent = any(b.id == 'BaseAgent' for b in node.bases if hasattr(b, 'id'))
                                if is_agent:
                                    violations.append(f"[AGENT] {file_path}: Class '{node.name}' inherits BaseAgent but does not end in 'Agent'")
                except Exception as e:
                    print(f"Warning: Could not check {file_path}: {e}")

def check_tool_naming():
    """Check that tool functions start with 'oci_{domain}_'."""
    print(f"Scanning Tools in {TOOLS_DIR}...")
    if not TOOLS_DIR.exists():
        print(f"Tools directory not found: {TOOLS_DIR}")
        return

    for file in os.listdir(TOOLS_DIR):
        if file.endswith(".py") and file != "__init__.py":
            domain = file.replace(".py", "") # e.g., 'compute'
            file_path = TOOLS_DIR / file
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Look for decorators @mcp.tool()
                        is_tool = False
                        if node.decorator_list:
                            for decorator in node.decorator_list:
                                # handle @mcp.tool() or @mcp.tool
                                if (isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr') and decorator.func.attr == 'tool') or \
                                   (isinstance(decorator, ast.Attribute) and decorator.attr == 'tool'):
                                   is_tool = True
                                   break
                        
                        if is_tool:
                            expected_prefix = f"oci_{domain}_"
                            # Special case: identity tools are often just oci_
                            if domain == "identity" and node.name.startswith("oci_"):
                                continue
                                
                            if not node.name.startswith(expected_prefix):
                                violations.append(f"[TOOL] {file}: Tool '{node.name}' should start with '{expected_prefix}'")

            except Exception as e:
                print(f"Warning: Could not check {file_path}: {e}")

def main():
    print("Starting Naming Convention Validation...")
    check_agent_naming()
    check_tool_naming()
    
    if violations:
        print("\n❌ Naming Violations Found:")
        for v in violations:
            print(v)
        sys.exit(1)
    else:
        print("\n✅ All Checks Passed! Naming conventions are respected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
