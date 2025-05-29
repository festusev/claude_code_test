#!/usr/bin/env python3
"""
Test suite for the Lisp to Python interpreter.
"""

import pytest
from lisp_to_python import (
    Tokenizer, Token, Parser, PythonGenerator, LispToPythonInterpreter,
    NumberNode, SymbolNode, ListNode, ASTNode
)


class TestTokenizer:
    def test_tokenize_basic_expression(self):
        tokenizer = Tokenizer("(+ 1 2)")
        tokens = tokenizer.tokenize()
        expected = [
            Token('LPAREN', '('),
            Token('SYMBOL', '+'),
            Token('NUMBER', '1'),
            Token('NUMBER', '2'),
            Token('RPAREN', ')')
        ]
        assert tokens == expected

    def test_tokenize_nested_expression(self):
        tokenizer = Tokenizer("(* (+ 1 2) 3)")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 9
        assert tokens[0].type == 'LPAREN'
        assert tokens[1].value == '*'
        assert tokens[2].type == 'LPAREN'

    def test_tokenize_negative_numbers(self):
        tokenizer = Tokenizer("(+ -1 -2.5)")
        tokens = tokenizer.tokenize()
        numbers = [t for t in tokens if t.type == 'NUMBER']
        assert numbers[0].value == '-1'
        assert numbers[1].value == '-2.5'

    def test_tokenize_symbols_with_special_chars(self):
        tokenizer = Tokenizer("(> <= !=)")
        tokens = tokenizer.tokenize()
        symbols = [t for t in tokens if t.type == 'SYMBOL']
        assert symbols[0].value == '>'
        assert symbols[1].value == '<='
        assert symbols[2].value == '!='

    def test_tokenize_comments(self):
        tokenizer = Tokenizer("(+ 1 2) ; this is a comment\n(* 3 4)")
        tokens = tokenizer.tokenize()
        # Comments should be skipped
        assert len(tokens) == 10  # Two complete expressions

    def test_tokenize_whitespace_handling(self):
        tokenizer = Tokenizer("  ( +   1    2  )  ")
        tokens = tokenizer.tokenize()
        expected = [
            Token('LPAREN', '('),
            Token('SYMBOL', '+'),
            Token('NUMBER', '1'),
            Token('NUMBER', '2'),
            Token('RPAREN', ')')
        ]
        assert tokens == expected


class TestParser:
    def test_parse_number(self):
        tokens = [Token('NUMBER', '42')]
        parser = Parser(tokens)
        ast = parser.parse()
        assert isinstance(ast, NumberNode)
        assert ast.value == 42

    def test_parse_float(self):
        tokens = [Token('NUMBER', '3.14')]
        parser = Parser(tokens)
        ast = parser.parse()
        assert isinstance(ast, NumberNode)
        assert ast.value == 3.14

    def test_parse_symbol(self):
        tokens = [Token('SYMBOL', 'x')]
        parser = Parser(tokens)
        ast = parser.parse()
        assert isinstance(ast, SymbolNode)
        assert ast.name == 'x'

    def test_parse_simple_list(self):
        tokens = [
            Token('LPAREN', '('),
            Token('SYMBOL', '+'),
            Token('NUMBER', '1'),
            Token('NUMBER', '2'),
            Token('RPAREN', ')')
        ]
        parser = Parser(tokens)
        ast = parser.parse()
        assert isinstance(ast, ListNode)
        assert len(ast.elements) == 3
        assert isinstance(ast.elements[0], SymbolNode)
        assert ast.elements[0].name == '+'

    def test_parse_nested_list(self):
        tokens = [
            Token('LPAREN', '('),
            Token('SYMBOL', '*'),
            Token('LPAREN', '('),
            Token('SYMBOL', '+'),
            Token('NUMBER', '1'),
            Token('NUMBER', '2'),
            Token('RPAREN', ')'),
            Token('NUMBER', '3'),
            Token('RPAREN', ')')
        ]
        parser = Parser(tokens)
        ast = parser.parse()
        assert isinstance(ast, ListNode)
        assert len(ast.elements) == 3
        assert isinstance(ast.elements[1], ListNode)

    def test_parse_empty_list(self):
        tokens = [Token('LPAREN', '('), Token('RPAREN', ')')]
        parser = Parser(tokens)
        ast = parser.parse()
        assert isinstance(ast, ListNode)
        assert len(ast.elements) == 0

    def test_parse_error_unmatched_paren(self):
        tokens = [Token('LPAREN', '('), Token('SYMBOL', '+')]
        parser = Parser(tokens)
        with pytest.raises(SyntaxError, match="Expected '\\)'"):
            parser.parse()

    def test_parse_error_unexpected_token(self):
        tokens = [Token('RPAREN', ')')]
        parser = Parser(tokens)
        with pytest.raises(SyntaxError, match="Unexpected token"):
            parser.parse()


class TestPythonGenerator:
    def test_generate_number(self):
        gen = PythonGenerator()
        node = NumberNode(42)
        result = gen.generate(node)
        assert result == "42"

    def test_generate_symbol(self):
        gen = PythonGenerator()
        node = SymbolNode("x")
        result = gen.generate(node)
        assert result == "x"

    def test_generate_addition(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("+"),
            NumberNode(1),
            NumberNode(2)
        ])
        result = gen.generate(node)
        assert result == "(1 + 2)"

    def test_generate_multiple_addition(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("+"),
            NumberNode(1),
            NumberNode(2),
            NumberNode(3)
        ])
        result = gen.generate(node)
        assert result == "((1 + 2) + 3)"

    def test_generate_nested_arithmetic(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("*"),
            ListNode([
                SymbolNode("+"),
                NumberNode(1),
                NumberNode(2)
            ]),
            NumberNode(3)
        ])
        result = gen.generate(node)
        assert result == "((1 + 2) * 3)"

    def test_generate_define(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("define"),
            SymbolNode("x"),
            NumberNode(10)
        ])
        result = gen.generate(node)
        assert result == "x = 10"

    def test_generate_defun(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("defun"),
            SymbolNode("square"),
            ListNode([SymbolNode("x")]),
            ListNode([
                SymbolNode("*"),
                SymbolNode("x"),
                SymbolNode("x")
            ])
        ])
        result = gen.generate(node)
        assert result == "def square(x): return (x * x)"

    def test_generate_if(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("if"),
            ListNode([
                SymbolNode(">"),
                SymbolNode("x"),
                NumberNode(5)
            ]),
            NumberNode(1),
            NumberNode(0)
        ])
        result = gen.generate(node)
        assert result == "(1 if (x > 5) else 0)"

    def test_generate_lambda(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("lambda"),
            ListNode([SymbolNode("x"), SymbolNode("y")]),
            ListNode([
                SymbolNode("+"),
                SymbolNode("x"),
                SymbolNode("y")
            ])
        ])
        result = gen.generate(node)
        assert result == "lambda x, y: (x + y)"

    def test_generate_function_call(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("square"),
            NumberNode(5)
        ])
        result = gen.generate(node)
        assert result == "square(5)"

    def test_generate_comparison_operators(self):
        gen = PythonGenerator()
        operators = {
            "=": "==",
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">="
        }
        
        for lisp_op, python_op in operators.items():
            node = ListNode([
                SymbolNode(lisp_op),
                SymbolNode("x"),
                NumberNode(5)
            ])
            result = gen.generate(node)
            assert result == f"(x {python_op} 5)"

    def test_generate_empty_list(self):
        gen = PythonGenerator()
        node = ListNode([])
        result = gen.generate(node)
        assert result == "[]"

    def test_generate_error_define_wrong_args(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("define"),
            SymbolNode("x")
        ])
        with pytest.raises(SyntaxError, match="define requires exactly 2 arguments"):
            gen.generate(node)

    def test_generate_error_defun_wrong_args(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("defun"),
            SymbolNode("func")
        ])
        with pytest.raises(SyntaxError, match="defun requires at least 3 arguments"):
            gen.generate(node)

    def test_generate_error_if_wrong_args(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("if"),
            SymbolNode("condition")
        ])
        with pytest.raises(SyntaxError, match="if requires exactly 3 arguments"):
            gen.generate(node)

    def test_generate_error_lambda_wrong_args(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("lambda"),
            ListNode([SymbolNode("x")])
        ])
        with pytest.raises(SyntaxError, match="lambda requires exactly 2 arguments"):
            gen.generate(node)

    def test_generate_error_binary_op_insufficient_args(self):
        gen = PythonGenerator()
        node = ListNode([
            SymbolNode("+"),
            NumberNode(1)
        ])
        with pytest.raises(SyntaxError, match="Unary operator \\+ not supported"):
            gen.generate(node)


class TestLispToPythonInterpreter:
    def test_interpret_basic_arithmetic(self):
        interpreter = LispToPythonInterpreter()
        
        test_cases = [
            ("(+ 1 2)", "(1 + 2)"),
            ("(- 5 3)", "(5 - 3)"),
            ("(* 4 6)", "(4 * 6)"),
            ("(/ 8 2)", "(8 / 2)"),
        ]
        
        for lisp_code, expected in test_cases:
            result = interpreter.interpret(lisp_code)
            assert result == expected

    def test_interpret_nested_expressions(self):
        interpreter = LispToPythonInterpreter()
        
        test_cases = [
            ("(+ (* 2 3) 4)", "((2 * 3) + 4)"),
            ("(* (+ 1 2) (- 4 3))", "((1 + 2) * (4 - 3))"),
            ("(/ (+ 10 5) (- 8 3))", "((10 + 5) / (8 - 3))"),
        ]
        
        for lisp_code, expected in test_cases:
            result = interpreter.interpret(lisp_code)
            assert result == expected

    def test_interpret_variable_definition(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("(define pi 3.14159)")
        assert result == "pi = 3.14159"

    def test_interpret_function_definition(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("(defun add (x y) (+ x y))")
        assert result == "def add(x, y): return (x + y)"

    def test_interpret_conditional(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("(if (> x 0) 1 -1)")
        assert result == "(1 if (x > 0) else -1)"

    def test_interpret_lambda(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("(lambda (x) (* x x))")
        assert result == "lambda x: (x * x)"

    def test_interpret_function_call(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("(sqrt 16)")
        assert result == "sqrt(16)"

    def test_interpret_comparison_operators(self):
        interpreter = LispToPythonInterpreter()
        
        test_cases = [
            ("(= x 5)", "(x == 5)"),
            ("(< x 10)", "(x < 10)"),
            ("(> y 0)", "(y > 0)"),
            ("(<= a b)", "(a <= b)"),
            ("(>= c d)", "(c >= d)"),
        ]
        
        for lisp_code, expected in test_cases:
            result = interpreter.interpret(lisp_code)
            assert result == expected

    def test_interpret_multiple_expressions(self):
        interpreter = LispToPythonInterpreter()
        lisp_code = "(define x 10) (define y 20)"
        results = interpreter.interpret_multiple(lisp_code)
        assert results == ["x = 10", "y = 20"]

    def test_interpret_empty_input(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("")
        assert result == ""

    def test_interpret_whitespace_only(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("   \n\t  ")
        assert result == ""

    def test_interpret_with_comments(self):
        interpreter = LispToPythonInterpreter()
        result = interpreter.interpret("(+ 1 2) ; This adds 1 and 2")
        assert result == "(1 + 2)"

    def test_interpret_error_handling(self):
        interpreter = LispToPythonInterpreter()
        
        with pytest.raises(SyntaxError):
            interpreter.interpret("(+ 1 2")  # Missing closing paren
        
        with pytest.raises(SyntaxError):
            interpreter.interpret("+ 1 2)")  # Missing opening paren
        
        with pytest.raises(SyntaxError):
            interpreter.interpret("(define x)")  # Missing value for define


class TestIntegration:
    def test_complex_expression(self):
        interpreter = LispToPythonInterpreter()
        lisp_code = "(defun factorial (n) (if (= n 0) 1 (* n (factorial (- n 1)))))"
        result = interpreter.interpret(lisp_code)
        expected = "def factorial(n): return (1 if (n == 0) else (n * factorial((n - 1))))"
        assert result == expected

    def test_mathematical_operations(self):
        interpreter = LispToPythonInterpreter()
        
        test_cases = [
            ("(+ 1 2 3 4 5)", "((((1 + 2) + 3) + 4) + 5)"),
            ("(* 2 3 4)", "((2 * 3) * 4)"),
            ("(- 10 3 2)", "((10 - 3) - 2)"),
            ("(/ 24 3 2)", "((24 / 3) / 2)"),
        ]
        
        for lisp_code, expected in test_cases:
            result = interpreter.interpret(lisp_code)
            assert result == expected

    def test_real_world_examples(self):
        interpreter = LispToPythonInterpreter()
        
        examples = [
            # Quadratic formula
            ("(defun quadratic (a b c) (/ (+ (- b) (sqrt (- (* b b) (* 4 a c)))) (* 2 a)))",
             "def quadratic(a, b, c): return (((-1 * b) + sqrt(((b * b) - ((4 * a) * c)))) / (2 * a))"),
            
            # Distance formula
            ("(defun distance (x1 y1 x2 y2) (sqrt (+ (* (- x2 x1) (- x2 x1)) (* (- y2 y1) (- y2 y1)))))",
             "def distance(x1, y1, x2, y2): return sqrt((((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1))))"),
            
            # Absolute value
            ("(defun abs (x) (if (< x 0) (- x) x))",
             "def abs(x): return ((-1 * x) if (x < 0) else x)"),
        ]
        
        for lisp_code, expected in examples:
            result = interpreter.interpret(lisp_code)
            assert result == expected