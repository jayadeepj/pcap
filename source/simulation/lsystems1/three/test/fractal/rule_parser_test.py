import pytest
from simulation.lsystems1.three.fractal.rule_parser import *

# Define test cases as tuples of (input_string, expected_result)
test_tokenize_data = [
    ("F(c)[+X]F(de)[-X]+X", ['F(c)', '[', '+', 'X', ']', 'F(de)', '[', '-', 'X', ']', '+', 'X']),
    ("Q(c)[+X]R(de)[-X]+X", ['Q(c)', '[', '+', 'X', ']', 'R(de)', '[', '-', 'X', ']', '+', 'X']),
    ("X", ['X']),
    ("F(2)[+X]F(14)[-X]+X", ['F(2)', '[', '+', 'X', ']', 'F(14)', '[', '-', 'X', ']', '+', 'X']),
    ("F(2)[+X]F(14)[-X]+X(30)", ['F(2)', '[', '+', 'X', ']', 'F(14)', '[', '-', 'X', ']', '+', 'X(30)']),
    ("F(2)[+(30)X/20]F(14)[-X]+X(30)",
     ['F(2)', '[', '+(30)', 'X', '/', '2', '0', ']', 'F(14)', '[', '-', 'X', ']', '+', 'X(30)']),
    ("N(w)F(l)[&(a0)B(l*r2,w*wr)]", ['N(w)', 'F(l)', '[', '&(a0)', 'B(l*r2,w*wr)', ']'])
]


@pytest.mark.parametrize("input_string, expected_result", test_tokenize_data)
def test_tokenize(input_string, expected_result):
    result = tokenize(input_string)
    assert result == expected_result


test_data = [
    ("F(10)", "F({})"),
    ("F(c+1, w*2)", "F({})"),
    ("F(c+1)X(w+1)", "F({})X({})")
]


@pytest.mark.parametrize("successor, expected_operand", test_data)
def test_extract_operand(successor, expected_operand):
    result = extract_operand(successor)
    assert result == expected_operand


test_check_format_data = [
    ("F(c)", "F(12.12)", True, {'c': 12.12}),
    ("F(c)", "F(200.3)", True, {'c': 200.3}),
    ("F(c,w)", "F(12.67,200.12)", True, {'c': 12.67, 'w': 200.12}),
    ("X(c,w)", "F(12.23, 200.43)", False, {}),
    ("A(s,w)", "A(100.0,5.0)", True, {'s': 100.0, 'w': 5.0})
]


# Define the test function using pytest.mark.parametrize
@pytest.mark.parametrize("pred, token, expected_result, expected_params", test_check_format_data)
def test_check_format(pred, token, expected_result, expected_params):
    result, params = check_format(pred, token)
    assert result == expected_result
    assert params == expected_params


test_extract_evals_data = [
    ("F(c+1)", ['c+1']),
    ("F(c+1, w*2)", ['c+1', 'w*2']),
    ("F(x^2 + 3*y, z-1)", ['x^2 + 3*y', 'z-1']),
    ("F(200, 10.0)", ['200', '10.0']),
    ("F(200, 10.0)N(32)", ['200', '10.0'])
]


# Define the test function using pytest.mark.parametrize
@pytest.mark.parametrize("successor, expected_evals", test_extract_evals_data)
def test_extract_evals(successor, expected_evals):
    result = extract_evals(successor)
    assert result == expected_evals


# Run tests using pytest
if __name__ == "__main__":
    pytest.main()
