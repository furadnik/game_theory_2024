import numpy as np


def evaluate(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Value of the row play when the row and column player use their respective strategies"""
    return row_strategy @ matrix @ column_strategy


def best_response_value_row(matrix: np.array, row_strategy: np.array) -> float:
    """Value of the row player when facing a best-responding column player"""
    return np.min(row_strategy @ matrix)


def best_response_value_column(matrix: np.array, column_strategy: np.array) -> float:
    """Value of the column player when facing a best-responding row player"""
    return -np.max(matrix @ column_strategy)
