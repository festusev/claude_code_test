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


class ASTNode:
    pass

@dataclass
class NumberNode(ASTNode):
    value: Union[int, float]

@dataclass 
class SymbolNode(ASTNode):
    name: str

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
    interpreter = LispToPythonInterpreter()
    
    # Example usage
    examples = [
        "(+ 1 2)",
        "(* (+ 1 2) 3)",
        "(define x 10)",
        "(defun square (x) (* x x))",
        "(if (> x 5) 1 0)",
        "(lambda (x y) (+ x y))",
        "(+ 1 2 3 4)",
    ]
    
    print("Lisp to Python Interpreter Examples:")
    print("=" * 40)
    
    for lisp_expr in examples:
        try:
            python_code = interpreter.interpret(lisp_expr)
            print(f"Lisp:   {lisp_expr}")
            print(f"Python: {python_code}")
            print()
        except Exception as e:
            print(f"Lisp:   {lisp_expr}")
            print(f"Error:  {e}")
            print()


if __name__ == "__main__":
    main()