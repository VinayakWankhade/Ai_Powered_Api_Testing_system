"""
Oracle module for advanced test validation and invariant checking.
"""

from .advanced_oracles import (
    AdvancedOracleSystem, 
    OracleType, 
    OracleResult, 
    InvariantType, 
    InvariantViolation
)

__all__ = [
    "AdvancedOracleSystem",
    "OracleType",
    "OracleResult", 
    "InvariantType",
    "InvariantViolation"
]
