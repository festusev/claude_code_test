#!/usr/bin/env python3
"""
Test suite for the Lisp to Python interpreter.
"""

import pytest
from lisp_to_python import (
    Tokenizer,
    Token,
    Parser,
    PythonGenerator,
    LispToPythonInterpreter,
    NumberNode,
    SymbolNode,
    ListNode,
    ASTNode,
)


class TestTokenizer:
    def test_tokenize_basic_expression(self):
        tokenizer = Tokenizer("(+ 1 2)")
        tokens = tokenizer.tokenize()
        expected = [
            Token("LPAREN", "("),
            Token("SYMBOL", "+"),
            Token("NUMBER", "1"),
            Token("NUMBER", "2"),
            Token("RPAREN", ")"),
        ]
        assert tokens == expected

    def test_tokenize_nested_expression(self):
        tokenizer = Tokenizer("(* (+ 1 2) 3)")
        tokens = tokenizer.tokenize()
        assert len(tokens) == 9
        assert tokens[0].type == "LPAREN"
        assert tokens[1].value == "*"
        assert tokens[2].type == "LPAREN"

    def test_tokenize_negative_numbers(self):
        tokenizer = Tokenizer("(+ -1 -2.5)")
        tokens = tokenizer.tokenize()
        numbers = [t for t in tokens if t.type == "NUMBER"]
        assert numbers[0].value == "-1"
        assert numbers[1].value == "-2.5"

    def test_tokenize_symbols_with_special_chars(self):
        tokenizer = Tokenizer("(> <= !=)")
        tokens = tokenizer.tokenize()
        symbols = [t for t in tokens if t.type == "SYMBOL"]
        assert symbols[0].value == ">"
        assert symbols[1].value == "<="
        assert symbols[2].value == "!="

    def test_tokenize_comments(self):
        tokenizer = Tokenizer("(+ 1 2) ; this is a comment\n(* 3 4)")
        tokens = tokenizer.tokenize()
        # Comments should be skipped
        assert len(tokens) == 10  # Two complete expressions

    def test_tokenize_whitespace_handling(self):
        tokenizer = Tokenizer("  ( +   1    2  )  ")
        tokens = tokenizer.tokenize()
        expected = [
            Token("LPAREN", "("),
            Token("SYMBOL", "+"),
            Token("NUMBER", "1"),
            Token("NUMBER", "2"),
            Token("RPAREN", ")"),
        ]
        assert tokens == expected
