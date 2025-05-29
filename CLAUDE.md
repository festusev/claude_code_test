# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
python -m pytest test_lisp_to_python.py          # Run all tests
python -m pytest test_lisp_to_python.py::TestTokenizer  # Run specific test class
python -m pytest -v                              # Run with verbose output
```

### Running the interpreter
```bash
python lisp_to_python.py example.lisp output.py   # Convert Lisp file to Python file
python -c "from lisp_to_python import LispToPythonInterpreter; print(LispToPythonInterpreter().interpret('(+ 1 2)'))"  # Interactive usage
```

### Type checking
```bash
mypy lisp_to_python.py test_lisp_to_python.py   # Type check both files
```

## Architecture

This is a **Lisp to Python transpiler** with a traditional 3-stage compilation pipeline:

1. **Tokenizer** (`Tokenizer` class) - Lexical analysis converting source text into tokens
2. **Parser** (`Parser` class) - Syntactic analysis building an AST from tokens  
3. **Code Generator** (`PythonGenerator` class) - Generates Python code from the AST

### Key Components

- **AST Nodes**: `NumberNode`, `SymbolNode`, `ListNode` represent the parsed Lisp syntax tree
- **LispToPythonInterpreter**: Main interface that orchestrates the compilation pipeline
- **Special Forms**: Handles Lisp constructs like `define`, `defun`, `if`, `lambda` with custom Python generation logic
- **Binary Operators**: Converts Lisp prefix notation `(+ 1 2)` to Python infix `(1 + 2)` with proper precedence handling

### Test Structure

Tests are organized by component (`TestTokenizer`, `TestParser`, `TestPythonGenerator`, `TestLispToPythonInterpreter`) plus integration tests (`TestIntegration`) that verify end-to-end transpilation of complex expressions.

## Code Review Requirements

- Every function should have reasonable test coverage when possible
- All tests must pass before code review approval  
- Test files should follow the existing naming convention (e.g., `test_*.py`)
- Tests should cover both happy path and edge cases where applicable
- GitHub CI workflow automatically runs tests and type checking on all PRs - both must pass before merging
- All changed code should be clean and readable
- This repository should use the ArgParse library exclusively for command-line argument parsing
- No linter errors should be thrown
- MyPy type checking must pass with no errors