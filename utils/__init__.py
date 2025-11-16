"""
AlphaTraderLab - Utility Functions

This package contains helper functions for training and evaluation.
"""

from .evaluation import (
    evaluate_agent,
    evaluate_random_agent,
    evaluate_buy_and_hold,
    compare_agents
)

__all__ = [
    'evaluate_agent',
    'evaluate_random_agent', 
    'evaluate_buy_and_hold',
    'compare_agents'
]
