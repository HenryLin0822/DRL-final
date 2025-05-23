"""
Karel DSL Parser and Executor

This module provides parsing and execution capabilities for Karel programs
written in the Domain Specific Language.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
import re
from .tokens import karel_tokens, TokenType


class ParseError(Exception):
    """Exception raised for parsing errors"""
    pass


class ExecutionError(Exception):
    """Exception raised for execution errors"""
    pass


class KarelProgram:
    """Represents a parsed Karel program"""
    
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.statements = []
        self.is_valid = False
        self.error_message = ""
    
    def __repr__(self):
        return f"KarelProgram(tokens={len(self.tokens)}, valid={self.is_valid})"


class KarelParser:
    """
    Karel DSL Parser
    
    Parses Karel programs and converts them into executable form.
    This is a simplified parser that handles the essential DSL constructs.
    """
    
    def __init__(self):
        self.tokens = karel_tokens
        self.max_execution_steps = 200
        self.call_stack_limit = 50
    
    def parse_string(self, program_string: str) -> KarelProgram:
        """Parse a program string into a KarelProgram"""
        try:
            token_list = self.tokens.string_to_tokens(program_string)
            return self.parse_tokens(token_list)
        except Exception as e:
            program = KarelProgram([])
            program.error_message = f"Parse error: {str(e)}"
            return program
    
    def parse_tokens(self, token_list: List[str]) -> KarelProgram:
        """Parse a list of tokens into a KarelProgram"""
        program = KarelProgram(token_list)
        
        try:
            # Validate tokens
            valid, error = self.tokens.validate_tokens(token_list)
            if not valid:
                program.error_message = error
                return program
            
            # Check program structure
            if not self._validate_program_structure(token_list):
                program.error_message = "Invalid program structure"
                return program
            
            # Validate bracket matching
            if not self._validate_bracket_matching(token_list):
                program.error_message = "Unmatched brackets"
                return program
            
            # Extract statement tokens (remove DEF run m( ... m))
            stmt_tokens = self.tokens.extract_statement(token_list)
            
            # Parse statements
            program.statements = self._parse_statements(stmt_tokens)
            program.is_valid = True
            
        except Exception as e:
            program.error_message = f"Parse error: {str(e)}"
        
        return program
    
    def _validate_program_structure(self, tokens: List[str]) -> bool:
        """Validate basic program structure"""
        if len(tokens) < 4:
            return False
        
        # Must start with DEF run m(
        if not self.tokens.is_valid_program_start(tokens):
            return False
        
        # Must end with m)
        if not self.tokens.is_valid_program_end(tokens):
            return False
        
        return True
    
    def _validate_bracket_matching(self, tokens: List[str]) -> bool:
        """Validate that all brackets are properly matched"""
        # Define bracket pairs
        bracket_pairs = {
            'm(': 'm)',
            'r(': 'r)',
            'i(': 'i)',
            'e(': 'e)',
            'c(': 'c)',
            'w(': 'w)'
        }
        
        # Stack for tracking open brackets
        bracket_stack = []
        
        for token in tokens:
            if token in bracket_pairs:  # Opening bracket
                bracket_stack.append(token)
            elif token in bracket_pairs.values():  # Closing bracket
                if not bracket_stack:
                    return False  # Closing bracket without opening
                
                last_open = bracket_stack[-1]
                expected_close = bracket_pairs[last_open]
                
                if token != expected_close:
                    return False  # Mismatched bracket
                
                bracket_stack.pop()
        
        # All brackets should be matched
        return len(bracket_stack) == 0
    
    def _parse_statements(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Parse statement tokens into executable form"""
        statements = []
        i = 0
        
        while i < len(tokens):
            try:
                stmt, consumed = self._parse_single_statement(tokens[i:])
                if stmt:
                    statements.append(stmt)
                i += consumed if consumed > 0 else 1
            except Exception as e:
                # If parsing fails, skip this token and continue
                i += 1
        
        return statements
    
    def _parse_single_statement(self, tokens: List[str]) -> Tuple[Optional[Dict], int]:
        """Parse a single statement, return (statement, tokens_consumed)"""
        if not tokens:
            return None, 0
        
        token = tokens[0]
        
        # Action statements
        if token in self.tokens.action_tokens:
            return {'type': 'action', 'action': token}, 1
        
        # Control flow statements
        elif token == 'REPEAT':
            return self._parse_repeat(tokens)
        elif token == 'IF':
            return self._parse_if(tokens)
        elif token == 'IFELSE':
            return self._parse_ifelse(tokens)
        elif token == 'WHILE':
            return self._parse_while(tokens)
        
        # Skip unknown tokens
        return None, 1
    
    def _parse_repeat(self, tokens: List[str]) -> Tuple[Optional[Dict], int]:
        """Parse REPEAT statement"""
        if len(tokens) < 5:  # REPEAT R=n r( ... r)
            return None, 1
        
        if tokens[0] != 'REPEAT' or tokens[2] != 'r(':
            return None, 1
        
        # Parse repeat count
        count_token = tokens[1]
        if count_token.startswith('R='):
            try:
                count = int(count_token[2:])
            except ValueError:
                return None, 1
        else:
            return None, 1
        
        # Find matching r)
        paren_count = 1
        i = 3
        while i < len(tokens) and paren_count > 0:
            if tokens[i] == 'r(':
                paren_count += 1
            elif tokens[i] == 'r)':
                paren_count -= 1
            i += 1
        
        if paren_count != 0:
            return None, 1  # Unmatched brackets
        
        # Parse body statements
        body_tokens = tokens[3:i-1]
        body_statements = self._parse_statements(body_tokens)
        
        return {
            'type': 'repeat',
            'count': count,
            'body': body_statements
        }, i
    
    def _parse_if(self, tokens: List[str]) -> Tuple[Optional[Dict], int]:
        """Parse IF statement"""
        if len(tokens) < 7:  # IF c( condition c) i( ... i)
            return None, 1
        
        if tokens[0] != 'IF' or tokens[1] != 'c(':
            return None, 1
        
        # Parse condition (handle 'not' case)
        condition_end = None
        negated = False
        condition = None
        
        # Look for c) to end condition
        for j in range(2, min(len(tokens), 6)):  # reasonable limit
            if tokens[j] == 'c)':
                condition_end = j
                break
        
        if condition_end is None:
            return None, 1
        
        # Extract condition tokens
        condition_tokens = tokens[2:condition_end]
        
        if len(condition_tokens) == 1:
            # Simple condition: c( condition c)
            condition = condition_tokens[0]
            negated = False
        elif len(condition_tokens) == 2 and condition_tokens[0] == 'not':
            # Negated condition: c( not condition c)
            condition = condition_tokens[1]
            negated = True
        else:
            return None, 1  # Invalid condition format
        
        # Check for i( after c)
        if condition_end + 1 >= len(tokens) or tokens[condition_end + 1] != 'i(':
            return None, 1
        
        start_body = condition_end + 2
        
        # Find matching i)
        paren_count = 1
        i = start_body
        while i < len(tokens) and paren_count > 0:
            if tokens[i] == 'i(':
                paren_count += 1
            elif tokens[i] == 'i)':
                paren_count -= 1
            i += 1
        
        if paren_count != 0:
            return None, 1  # Unmatched brackets
        
        # Parse body statements
        body_tokens = tokens[start_body:i-1]
        body_statements = self._parse_statements(body_tokens)
        
        return {
            'type': 'if',
            'condition': condition,
            'negated': negated,
            'body': body_statements
        }, i
    
    def _parse_ifelse(self, tokens: List[str]) -> Tuple[Optional[Dict], int]:
        """Parse IFELSE statement"""
        # This is a simplified version - full implementation would handle
        # IFELSE c( condition c) i( ... i) ELSE e( ... e)
        return None, 1
    
    def _parse_while(self, tokens: List[str]) -> Tuple[Optional[Dict], int]:
        """Parse WHILE statement"""
        if len(tokens) < 7:  # WHILE c( condition c) w( ... w)
            return None, 1
        
        if tokens[0] != 'WHILE' or tokens[1] != 'c(':
            return None, 1
        
        # Parse condition (handle 'not' case)
        condition_end = None
        negated = False
        condition = None
        
        # Look for c) to end condition
        for j in range(2, min(len(tokens), 6)):  # reasonable limit
            if tokens[j] == 'c)':
                condition_end = j
                break
        
        if condition_end is None:
            return None, 1
        
        # Extract condition tokens
        condition_tokens = tokens[2:condition_end]
        
        if len(condition_tokens) == 1:
            # Simple condition: c( condition c)
            condition = condition_tokens[0]
            negated = False
        elif len(condition_tokens) == 2 and condition_tokens[0] == 'not':
            # Negated condition: c( not condition c)
            condition = condition_tokens[1]
            negated = True
        else:
            return None, 1  # Invalid condition format
        
        # Check for w( after c)
        if condition_end + 1 >= len(tokens) or tokens[condition_end + 1] != 'w(':
            return None, 1
        
        start_body = condition_end + 2
        
        # Find matching w)
        paren_count = 1
        i = start_body
        while i < len(tokens) and paren_count > 0:
            if tokens[i] == 'w(':
                paren_count += 1
            elif tokens[i] == 'w)':
                paren_count -= 1
            i += 1
        
        if paren_count != 0:
            return None, 1  # Unmatched brackets
        
        # Parse body statements
        body_tokens = tokens[start_body:i-1]
        body_statements = self._parse_statements(body_tokens)
        
        return {
            'type': 'while',
            'condition': condition,
            'negated': negated,
            'body': body_statements
        }, i


class KarelExecutor:
    """
    Karel Program Executor
    
    Executes parsed Karel programs on a Karel world.
    """
    
    def __init__(self):
        self.max_steps = 200
        self.step_count = 0
        self.call_depth = 0
        self.max_call_depth = 50
    
    def execute(self, program: KarelProgram, karel_world) -> List[np.ndarray]:
        """
        Execute a Karel program on a world
        
        Args:
            program: Parsed Karel program
            karel_world: Karel world instance
            
        Returns:
            List of states representing execution trace
        """
        if not program.is_valid:
            raise ExecutionError(f"Cannot execute invalid program: {program.error_message}")
        
        # Reset execution state
        self.step_count = 0
        self.call_depth = 0
        
        # Initialize execution trace with starting state
        execution_trace = [karel_world.get_state().copy()]
        
        try:
            # Execute all statements
            for statement in program.statements:
                self._execute_statement(statement, karel_world, execution_trace)
                
                # Check for timeout
                if self.step_count >= self.max_steps:
                    break
                    
                # Check if world is done
                if karel_world.done:
                    break
        
        except ExecutionError:
            raise
        except Exception as e:
            raise ExecutionError(f"Execution error: {str(e)}")
        
        return execution_trace
    
    def _execute_statement(self, statement: Dict[str, Any], karel_world, trace: List[np.ndarray]):
        """Execute a single statement"""
        if self.step_count >= self.max_steps:
            return
        
        if self.call_depth >= self.max_call_depth:
            raise ExecutionError("Call stack overflow")
        
        stmt_type = statement['type']
        
        if stmt_type == 'action':
            self._execute_action(statement['action'], karel_world, trace)
        
        elif stmt_type == 'repeat':
            self._execute_repeat(statement, karel_world, trace)
        
        elif stmt_type == 'if':
            self._execute_if(statement, karel_world, trace)
        
        elif stmt_type == 'while':
            self._execute_while(statement, karel_world, trace)
    
    def _execute_action(self, action: str, karel_world, trace: List[np.ndarray]):
        """Execute a primitive action"""
        if self.step_count >= self.max_steps:
            return
        
        # Map DSL actions to Karel world actions
        action_map = {
            'move': 0,
            'turnLeft': 1,
            'turnRight': 2,
            'pickMarker': 3,
            'putMarker': 4
        }
        
        if action in action_map:
            karel_action = action_map[action]
            _, _, done, _ = karel_world.step(karel_action)
            trace.append(karel_world.get_state().copy())
            self.step_count += 1
    
    def _execute_repeat(self, statement: Dict[str, Any], karel_world, trace: List[np.ndarray]):
        """Execute REPEAT statement"""
        count = statement['count']
        body = statement['body']
        
        self.call_depth += 1
        try:
            for _ in range(count):
                if self.step_count >= self.max_steps or karel_world.done:
                    break
                
                for stmt in body:
                    self._execute_statement(stmt, karel_world, trace)
                    if self.step_count >= self.max_steps or karel_world.done:
                        break
        finally:
            self.call_depth -= 1
    
    def _execute_if(self, statement: Dict[str, Any], karel_world, trace: List[np.ndarray]):
        """Execute IF statement"""
        condition = statement['condition']
        negated = statement['negated']
        body = statement['body']
        
        # Evaluate condition
        condition_result = self._evaluate_condition(condition, karel_world)
        if negated:
            condition_result = not condition_result
        
        if condition_result:
            self.call_depth += 1
            try:
                for stmt in body:
                    self._execute_statement(stmt, karel_world, trace)
                    if self.step_count >= self.max_steps or karel_world.done:
                        break
            finally:
                self.call_depth -= 1
    
    def _execute_while(self, statement: Dict[str, Any], karel_world, trace: List[np.ndarray]):
        """Execute WHILE statement"""
        condition = statement['condition']
        negated = statement['negated']
        body = statement['body']
        
        self.call_depth += 1
        try:
            while self.step_count < self.max_steps and not karel_world.done:
                # Evaluate condition
                condition_result = self._evaluate_condition(condition, karel_world)
                if negated:
                    condition_result = not condition_result
                
                if not condition_result:
                    break
                
                # Execute body
                for stmt in body:
                    self._execute_statement(stmt, karel_world, trace)
                    if self.step_count >= self.max_steps or karel_world.done:
                        break
        finally:
            self.call_depth -= 1
    
    def _evaluate_condition(self, condition: str, karel_world) -> bool:
        """Evaluate a condition in the Karel world"""
        condition_map = {
            'frontIsClear': karel_world.front_is_clear,
            'leftIsClear': karel_world.left_is_clear,
            'rightIsClear': karel_world.right_is_clear,
            'markersPresent': karel_world.marker_present,
            'noMarkersPresent': karel_world.no_marker_present,
        }
        
        if condition in condition_map:
            return condition_map[condition]()
        else:
            return False  # Unknown condition defaults to False


class KarelDSLParser:
    """
    Main Karel DSL interface
    
    Combines parsing and execution in a simple interface.
    """
    
    def __init__(self):
        self.parser = KarelParser()
        self.executor = KarelExecutor()
        self.tokens = karel_tokens
    
    def parse(self, program_string: str) -> KarelProgram:
        """Parse a program string"""
        return self.parser.parse_string(program_string)
    
    def execute(self, program_string: str, karel_world) -> List[np.ndarray]:
        """Parse and execute a program string"""
        program = self.parser.parse_string(program_string)
        return self.executor.execute(program, karel_world)
    
    def execute_tokens(self, tokens: List[str], karel_world) -> List[np.ndarray]:
        """Parse and execute a token list"""
        program = self.parser.parse_tokens(tokens)
        return self.executor.execute(program, karel_world)
    
    def execute_indices(self, indices: List[int], karel_world) -> List[np.ndarray]:
        """Parse and execute from token indices"""
        # Convert indices to tokens
        tokens = self.tokens.indices_to_tokens(indices)
        # Remove padding
        tokens = self.tokens.filter_padding(tokens)
        return self.execute_tokens(tokens, karel_world)
    
    def tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string"""
        return self.tokens.tokens_to_string(tokens)
    
    def string_to_tokens(self, program_string: str) -> List[str]:
        """Convert string to tokens"""
        return self.tokens.string_to_tokens(program_string)
    
    def validate_program(self, program_string: str) -> Tuple[bool, str]:
        """Validate a program string"""
        try:
            program = self.parser.parse_string(program_string)
            if program.is_valid:
                return True, "Program is valid"
            else:
                return False, program.error_message
        except Exception as e:
            return False, f"Validation error: {str(e)}"


# Create global instance
karel_dsl = KarelDSLParser()

# Export convenience functions
def parse_program(program_string: str) -> KarelProgram:
    """Parse a Karel program string"""
    return karel_dsl.parse(program_string)

def execute_program(program_string: str, karel_world) -> List[np.ndarray]:
    """Execute a Karel program string"""
    return karel_dsl.execute(program_string, karel_world)