#!/usr/bin/env python3
"""
Simple Lisp to Python Interpreter

Supports basic Lisp expressions and converts them to Python code.
Examples:
- (+ 1 2) -> (1 + 2)
- (* (+ 1 2) 3) -> ((1 + 2) * 3)
- (define x 10) -> x = 10
- (defun square (x) (* x x)) -> def square(x): return (x * x)
"""

import re
import sys
from typing import List, Union, Any
from dataclasses import dataclass


@dataclass
class Token:
    type: str
    value: str
    

class Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        
    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break
                
            char = self.text[self.pos]
            
            if char == '(':
                tokens.append(Token('LPAREN', '('))
                self.pos += 1
            elif char == ')':
                tokens.append(Token('RPAREN', ')'))
                self.pos += 1
            elif char == ';':
                self._skip_comment()
            elif char == '"':
                tokens.append(self._read_string())
            elif char.isdigit() or (char == '-' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
                tokens.append(self._read_number())
            elif char.isalpha() or char in '+-*/=<>!':
                tokens.append(self._read_symbol())
            else:
                self.pos += 1
                
        return tokens
    
    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
            
    def _skip_comment(self):
        while self.pos < len(self.text) and self.text[self.pos] != '\n':
            self.pos += 1
            
    def _read_number(self) -> Token:
        start = self.pos
        if self.text[self.pos] == '-':
            self.pos += 1
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        return Token('NUMBER', self.text[start:self.pos])
    
    def _read_symbol(self) -> Token:
        start = self.pos
        while (self.pos < len(self.text) and 
               (self.text[self.pos].isalnum() or self.text[self.pos] in '+-*/=<>!?-_')):
            self.pos += 1
        return Token('SYMBOL', self.text[start:self.pos])
    
    def _read_string(self) -> Token:
        start = self.pos
        self.pos += 1  # Skip opening quote
        while self.pos < len(self.text) and self.text[self.pos] != '"':
            if self.text[self.pos] == '\\' and self.pos + 1 < len(self.text):
                self.pos += 2  # Skip escape sequence
            else:
                self.pos += 1
        if self.pos >= len(self.text):
            raise SyntaxError("Unterminated string literal")
        self.pos += 1  # Skip closing quote
        return Token('STRING', self.text[start:self.pos])


class ASTNode:
    pass

@dataclass
class NumberNode(ASTNode):
    value: Union[int, float]

@dataclass 
class SymbolNode(ASTNode):
    name: str

@dataclass
class StringNode(ASTNode):
    value: str

@dataclass
class ListNode(ASTNode):
    elements: List[ASTNode]


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        
    def parse(self) -> ASTNode:
        return self._parse_expression()
    
    def _parse_expression(self) -> ASTNode:
        if self.pos >= len(self.tokens):
            raise SyntaxError("Unexpected end of input")
            
        token = self.tokens[self.pos]
        
        if token.type == 'NUMBER':
            self.pos += 1
            value = float(token.value) if '.' in token.value else int(token.value)
            return NumberNode(value)
        elif token.type == 'STRING':
            self.pos += 1
            return StringNode(token.value)
        elif token.type == 'SYMBOL':
            self.pos += 1
            return SymbolNode(token.value)
        elif token.type == 'LPAREN':
            return self._parse_list()
        else:
            raise SyntaxError(f"Unexpected token: {token.value}")
    
    def _parse_list(self) -> ListNode:
        if self.tokens[self.pos].type != 'LPAREN':
            raise SyntaxError("Expected '('")
        
        self.pos += 1  # consume '('
        elements = []
        
        while self.pos < len(self.tokens) and self.tokens[self.pos].type != 'RPAREN':
            elements.append(self._parse_expression())
        
        if self.pos >= len(self.tokens) or self.tokens[self.pos].type != 'RPAREN':
            raise SyntaxError("Expected ')'")
        
        self.pos += 1  # consume ')'
        return ListNode(elements)


class PythonGenerator:
    def __init__(self):
        self.indent_level = 0
        
    def generate(self, node: ASTNode) -> str:
        if isinstance(node, NumberNode):
            return str(node.value)
        elif isinstance(node, SymbolNode):
            return node.name
        elif isinstance(node, StringNode):
            return node.value  # Keep the quotes
        elif isinstance(node, ListNode):
            return self._generate_list(node)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
    
    def _generate_list(self, node: ListNode) -> str:
        if not node.elements:
            return "[]"
        
        first = node.elements[0]
        if isinstance(first, SymbolNode):
            if first.name == 'define':
                return self._generate_define(node.elements[1:])
            elif first.name == 'defun':
                return self._generate_defun(node.elements[1:])
            elif first.name in ['+', '-', '*', '/', '=', '<', '>', '<=', '>=']:
                return self._generate_binary_op(first.name, node.elements[1:])
            elif first.name == 'if':
                return self._generate_if(node.elements[1:])
            elif first.name == 'lambda':
                return self._generate_lambda(node.elements[1:])
            elif first.name == 'let':
                return self._generate_let(node.elements[1:])
            elif first.name == 'cond':
                return self._generate_cond(node.elements[1:])
            elif first.name in ['car', 'cdr', 'cons']:
                return self._generate_list_op(first.name, node.elements[1:])
            elif first.name in ['length', 'append']:
                return self._generate_list_builtin(first.name, node.elements[1:])
            else:
                # Function call
                func_name = self.generate(first)
                args = [self.generate(arg) for arg in node.elements[1:]]
                return f"{func_name}({', '.join(args)})"
        else:
            # List of expressions
            return f"[{', '.join(self.generate(elem) for elem in node.elements)}]"
    
    def _generate_define(self, args: List[ASTNode]) -> str:
        if len(args) != 2:
            raise SyntaxError("define requires exactly 2 arguments")
        var_name = self.generate(args[0])
        value = self.generate(args[1])
        return f"{var_name} = {value}"
    
    def _generate_defun(self, args: List[ASTNode]) -> str:
        if len(args) < 3:
            raise SyntaxError("defun requires at least 3 arguments")
        
        func_name = self.generate(args[0])
        if not isinstance(args[1], ListNode):
            raise SyntaxError("Function parameters must be a list")
        
        params = [self.generate(param) for param in args[1].elements]
        body = self.generate(args[2])
        
        return f"def {func_name}({', '.join(params)}): return {body}"
    
    def _generate_binary_op(self, op: str, args: List[ASTNode]) -> str:
        if op == '=':
            op = '=='
        
        # Handle unary operators (like unary minus)
        if len(args) == 1:
            if op == '-':
                return f"(-1 * {self.generate(args[0])})"
            else:
                raise SyntaxError(f"Unary operator {op} not supported")
        
        if len(args) < 2:
            raise SyntaxError(f"Operator {op} requires at least 2 arguments")
        
        if len(args) == 2:
            left = self.generate(args[0])
            right = self.generate(args[1])
            return f"({left} {op} {right})"
        else:
            # Multiple arguments: (+ 1 2 3) -> (((1 + 2) + 3))
            result = f"({self.generate(args[0])} {op} {self.generate(args[1])})"
            for arg in args[2:]:
                result = f"({result} {op} {self.generate(arg)})"
            return result
    
    def _generate_if(self, args: List[ASTNode]) -> str:
        if len(args) != 3:
            raise SyntaxError("if requires exactly 3 arguments")
        
        condition = self.generate(args[0])
        then_expr = self.generate(args[1])
        else_expr = self.generate(args[2])
        
        return f"({then_expr} if {condition} else {else_expr})"
    
    def _generate_lambda(self, args: List[ASTNode]) -> str:
        if len(args) != 2:
            raise SyntaxError("lambda requires exactly 2 arguments")
        
        if not isinstance(args[0], ListNode):
            raise SyntaxError("Lambda parameters must be a list")
        
        params = [self.generate(param) for param in args[0].elements]
        body = self.generate(args[1])
        
        return f"lambda {', '.join(params)}: {body}"
    
    def _generate_let(self, args: List[ASTNode]) -> str:
        if len(args) != 2:
            raise SyntaxError("let requires exactly 2 arguments")
        
        if not isinstance(args[0], ListNode):
            raise SyntaxError("let bindings must be a list")
        
        bindings = args[0].elements
        body = self.generate(args[1])
        
        # Convert let bindings to lambda application: (let ((x 1) (y 2)) body) -> ((lambda (x y) body) 1 2)
        if not bindings:
            return body
        
        # Extract variable names and values
        var_names = []
        values = []
        for binding in bindings:
            if not isinstance(binding, ListNode) or len(binding.elements) != 2:
                raise SyntaxError("let binding must be a list of [var value]")
            var_names.append(self.generate(binding.elements[0]))
            values.append(self.generate(binding.elements[1]))
        
        params = ', '.join(var_names)
        args_str = ', '.join(values)
        return f"(lambda {params}: {body})({args_str})"
    
    def _generate_cond(self, args: List[ASTNode]) -> str:
        if len(args) == 0:
            raise SyntaxError("cond requires at least one clause")
        
        # Build nested if-else expression
        result = "None"
        for clause in reversed(args):
            if not isinstance(clause, ListNode) or len(clause.elements) != 2:
                raise SyntaxError("cond clause must be [condition expression]")
            
            condition = self.generate(clause.elements[0])
            expression = self.generate(clause.elements[1])
            
            result = f"({expression} if {condition} else {result})"
        
        return result
    
    def _generate_list_op(self, op: str, args: List[ASTNode]) -> str:
        if op == 'car':
            if len(args) != 1:
                raise SyntaxError("car requires exactly 1 argument")
            return f"{self.generate(args[0])}[0]"
        elif op == 'cdr':
            if len(args) != 1:
                raise SyntaxError("cdr requires exactly 1 argument")
            return f"{self.generate(args[0])}[1:]"
        elif op == 'cons':
            if len(args) != 2:
                raise SyntaxError("cons requires exactly 2 arguments")
            return f"[{self.generate(args[0])}] + {self.generate(args[1])}"
        else:
            raise ValueError(f"Unknown list operation: {op}")
    
    def _generate_list_builtin(self, op: str, args: List[ASTNode]) -> str:
        if op == 'length':
            if len(args) != 1:
                raise SyntaxError("length requires exactly 1 argument")
            return f"len({self.generate(args[0])})"
        elif op == 'append':
            if len(args) != 2:
                raise SyntaxError("append requires exactly 2 arguments")
            return f"{self.generate(args[0])} + {self.generate(args[1])}"
        else:
            raise ValueError(f"Unknown list builtin: {op}")


class LispToPythonInterpreter:
    def __init__(self):
        self.generator = PythonGenerator()
    
    def interpret(self, lisp_code: str) -> str:
        """Convert Lisp code to Python code."""
        tokenizer = Tokenizer(lisp_code)
        tokens = tokenizer.tokenize()
        
        if not tokens:
            return ""
        
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Check if all tokens were consumed
        if parser.pos < len(tokens):
            raise SyntaxError(f"Unexpected tokens after expression: {[t.value for t in tokens[parser.pos:]]}")
        
        return self.generator.generate(ast)
    
    def interpret_multiple(self, lisp_code: str) -> List[str]:
        """Convert multiple Lisp expressions to Python code."""
        results = []
        tokenizer = Tokenizer(lisp_code)
        tokens = tokenizer.tokenize()
        
        parser = Parser(tokens)
        while parser.pos < len(tokens):
            ast = parser.parse()
            results.append(self.generator.generate(ast))
        
        return results


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Convert Lisp files to Python')
    parser.add_argument('input_file', help='Input Lisp file (.lisp extension recommended)')
    parser.add_argument('output_file', help='Output Python file (.py extension recommended)')
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    
    # Input validation
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    if not input_file.lower().endswith('.lisp'):
        print(f"Warning: Input file '{input_file}' does not have .lisp extension")
    
    if not output_file.lower().endswith('.py'):
        print(f"Warning: Output file '{output_file}' does not have .py extension")
    
    # Check if output directory exists and is writable
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist")
        sys.exit(1)
    
    if output_dir and not os.access(output_dir, os.W_OK):
        print(f"Error: Output directory '{output_dir}' is not writable")
        sys.exit(1)
    
    interpreter = LispToPythonInterpreter()
    
    try:
        with open(input_file, 'r') as f:
            lisp_code = f.read()
        
        # Handle multiple expressions
        results = interpreter.interpret_multiple(lisp_code)
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(result + '\n')
        
        print(f"Converted {len(results)} expressions from {input_file} to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: Permission denied - {e}")
        sys.exit(1)
    except SyntaxError as e:
        print(f"Error: Lisp syntax error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error occurred - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()