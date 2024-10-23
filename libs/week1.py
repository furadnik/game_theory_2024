from typing import Iterable

import numpy as np

from .utils import Action, Value


def evaluate(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> Value:
    """Value of the row play when the row and column player use their respective strategies"""
    return row_strategy @ matrix @ column_strategy


def best_response_value_row(matrix: np.array, row_strategy: np.array) -> Value:
    """Value of the row player when facing a best-responding column player"""
    return np.min(row_strategy @ matrix)


def best_response_value_column(matrix: np.array, column_strategy: np.array) -> Value:
    """Value of the column player when facing a best-responding row player"""
    return -np.max(matrix @ column_strategy)


def best_response_action_row(matrix: np.array, row_strategy: np.array) -> Action:
    """br action of the column player when given a row strategy"""
    return np.argmin(row_strategy @ matrix)


def best_response_action_column(matrix: np.array, column_strategy: np.array) -> Action:
    """br action of the row player when given a column strategy"""
    return np.argmax(matrix @ column_strategy)


def find_row_dominated(matrix: np.ndarray) -> Iterable[tuple[Action, Action]]:
    """Find row dominated actions."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i == j:
                continue
            if np.all(matrix[j] >= matrix[i]):
                yield i, j


def find_col_dominated(matrix: np.ndarray) -> Iterable[tuple[Action, Action]]:
    """Find column dominated actions."""
    return find_row_dominated(matrix.T)


def elimination_of_dominated(matrix_player1: np.array, matrix_player2: np.array) -> tuple[np.array, np.array]:
    """Eliminate dominated strategies."""
    while True:
        changed = False
        for action, _ in find_row_dominated(matrix_player1):
            keep_indices = [x for x in range(matrix_player1.shape[0]) if x != action]
            matrix_player1 = matrix_player1[keep_indices]
            matrix_player2 = matrix_player2[keep_indices]
            changed = True
        for action, _ in find_col_dominated(matrix_player2):
            print(action)
            keep_indices = [x for x in range(matrix_player2.shape[1]) if x != action]
            matrix_player1 = matrix_player1[:, keep_indices]
            matrix_player2 = matrix_player2[:, keep_indices]
            changed = True
        if not changed:
            return matrix_player1, matrix_player2
