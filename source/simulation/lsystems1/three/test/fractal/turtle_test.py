import pytest
from simulation.lsystems1.three.fractal.turtle import *


@pytest.fixture
def create_points():
    point1 = Point(0, 0, 0)
    point2 = Point(3, 4, 0)
    point3 = Point(0, 5, 12)
    point4 = Point(-2, 1, 5)
    return point1, point2, point3, point4


def test_distance_between_points(create_points):
    point1, point2, point3, point4 = create_points
    distance = point1.distance_to(point2)
    assert math.isclose(distance, 5.0, rel_tol=1e-3)


def test_distance_point1_point3(create_points):
    point1, point2, point3, point4 = create_points
    distance = point1.distance_to(point3)
    assert math.isclose(distance, 13.0, rel_tol=1e-3)


def test_distance_point2_point3(create_points):
    point1, point2, point3, point4 = create_points
    distance = point2.distance_to(point3)
    assert math.isclose(distance, 12.409674, rel_tol=1e-3)


def test_distance_point3_point4(create_points):
    point1, point2, point3, point4 = create_points
    distance = point3.distance_to(point4)
    assert math.isclose(distance, 8.306624, rel_tol=1e-3)


def test_point_equality():
    point1 = Point(1, 2, 3)
    point2 = Point(1, 2, 3)
    assert point1 == point2
    assert hash(point1) == hash(point2)


def test_line_equality_with_different_points():
    point1 = Point(1, 2, 3)
    point2 = Point(1, 2, 3)
    line1 = TurtleLine(point1, "h", 2.5)
    line2 = TurtleLine(point2, "l", 1.0)
    assert line1 == line2


def test_line1_not_equals_line3():
    point1 = Point(1, 2, 3)
    point2 = Point(4, 5, 6)
    line1 = TurtleLine(point1, "h", 2.5)
    line3 = TurtleLine(point2, "u", 3.0)
    assert line1 != line3


# Run tests using pytest
if __name__ == "__main__":
    pytest.main()  # Run pytest (all functions with test_) with verbose output
