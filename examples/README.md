# Lisp to Python Examples

This directory contains example Lisp files demonstrating various features supported by the Lisp to Python transpiler.

## Examples

### 1. arithmetic.lisp
Basic arithmetic operations including addition, subtraction, multiplication, division, and nested expressions.

**Run:**
```bash
python lisp_to_python.py examples/arithmetic.lisp output_arithmetic.py
python output_arithmetic.py
```

### 2. functions.lisp
Function definitions using `defun` and function calls, including recursive functions like factorial.

**Run:**
```bash
python lisp_to_python.py examples/functions.lisp output_functions.py
python output_functions.py
```

### 3. conditionals.lisp
Conditional expressions using `if`, variable definitions with `define`, and nested conditionals.

**Run:**
```bash
python lisp_to_python.py examples/conditionals.lisp output_conditionals.py
python output_conditionals.py
```

### 4. lambda.lisp
Lambda expressions, higher-order functions, and anonymous function usage.

**Run:**
```bash
python lisp_to_python.py examples/lambda.lisp output_lambda.py
python output_lambda.py
```

## Quick Test

To quickly test all examples:

```bash
# Run each example
for file in examples/*.lisp; do
    echo "Testing $file..."
    python lisp_to_python.py "$file" "output_$(basename "$file" .lisp).py"
    python "output_$(basename "$file" .lisp).py"
    echo "---"
done
```

## Interactive Usage

You can also test individual expressions interactively:

```bash
python -c "from lisp_to_python import LispToPythonInterpreter; print(LispToPythonInterpreter().interpret('(+ 1 2)'))"
```