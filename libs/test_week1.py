import numpy as np
import pytest

import libs.week1 as week1


@pytest.fixture
def random_matrix() -> tuple[np.array, np.array, np.array, int, int]:
    m, n = np.random.randint(1, 100), np.random.randint(1, 100)
    matrix = np.random.rand(m, n) * 200 + 100
    row_strategy = np.random.rand(m)
    column_strategy = np.random.rand(n)
    return matrix, row_strategy, column_strategy, m, n


def test_week1():
    matrix = np.array([[0, 1, -1],
                       [-1, 0, 1],
                       [1, -1, 0]])
    row_strategy = np.array([[0.1, 0.2, 0.7]])
    column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()

    row_value = week1.evaluate(matrix=matrix, row_strategy=row_strategy, column_strategy=column_strategy)
    assert row_value == pytest.approx(0.08)

    br_value_row = week1.best_response_value_row(matrix=matrix, row_strategy=row_strategy)
    br_value_column = week1.best_response_value_column(matrix=matrix, column_strategy=column_strategy)
    assert br_value_row == pytest.approx(-0.6)
    assert br_value_column == pytest.approx(-0.2)


@pytest.mark.parametrize("rep", range(100))
def test_br_value_eq_strategy(rep, random_matrix):
    matrix, row_strategy, column_strategy, m, n = random_matrix
    br_value_row = week1.best_response_value_row(matrix=matrix, row_strategy=row_strategy)
    br_value_column = week1.best_response_value_column(matrix=matrix, column_strategy=column_strategy)

    br_action_row = week1.best_response_action_row(matrix=matrix, row_strategy=row_strategy)
    br_action_column = week1.best_response_action_column(matrix=matrix, column_strategy=column_strategy)
    br_strategy_row = np.eye(n)[br_action_row]
    br_strategy_column = np.eye(m)[br_action_column]
    assert pytest.approx(br_value_row) == week1.evaluate(matrix, row_strategy, br_strategy_row)
    assert pytest.approx(br_value_column) == -week1.evaluate(matrix, br_strategy_column, column_strategy)


def test_dominated_actions():
    matrix = np.array([[0, 1, -1],
                       [-1, 0, 1],
                       [1, -1, 0]])
    assert list(week1.find_row_dominated(matrix)) == []
    matrix = np.array([[0, 0, -1],
                       [-1, 0, 1],
                       [1, 1, 0]])
    assert list(week1.find_row_dominated(matrix)) == [(0, 2)]
    matrix = np.array([[0, 1, -1],
                       [-1, 0, 1],
                       [1, -1, 0]]).T
    assert list(week1.find_col_dominated(matrix)) == []
    matrix = np.array([[0, 0, -1],
                       [-1, 0, 1],
                       [1, 1, 0]]).T
    assert list(week1.find_col_dominated(matrix)) == [(0, 2)]


def test_elimination():
    matrix_1 = np.array([[13, 1, 7],
                         [4, 3, 6],
                         [-1, 2, 8]])
    matrix_2 = np.array([[3, 4, 3],
                         [1, 3, 2],
                         [9, 8, -1]])

    new_1, new_2 = week1.elimination_of_dominated(matrix_1, matrix_2)
    assert np.all(new_1 == np.array([3])), new_1
    assert np.all(new_2 == np.array([3])), new_1
