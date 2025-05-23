"""
Karel Domain Specific Language Package

This package provides a clean, simplified implementation of the Karel DSL
for use with the HPRL framework.
"""

from .tokens import (
    karel_tokens,
    TokenType,
    string_to_indices,
    indices_to_string,
    get_vocab_size,
    get_padding_index,
    pad_sequence
)

from .parser import (
    KarelDSLParser,
    KarelProgram,
    ParseError,
    ExecutionError,
    karel_dsl as default_parser
)

from .karel_dsl import (
    KarelDSL,
    get_DSL_option_v2,
    default_dsl
)

# Main exports for easy importing
__all__ = [
    # Token management
    'karel_tokens',
    'TokenType',
    'string_to_indices',
    'indices_to_string',
    'get_vocab_size',
    'get_padding_index',
    'pad_sequence',
    
    # Parsing and execution
    'KarelDSLParser',
    'KarelProgram',
    'ParseError',
    'ExecutionError',
    'default_parser',
    
    # Main DSL interface
    'KarelDSL',
    'get_DSL_option_v2',
    'default_dsl'
]

# Version info
__version__ = '1.0.0'