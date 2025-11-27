import math
import random

from typing import List
from anytree import NodeMixin
from simulation.lsystems1.three.fractal import rule_parser as parser
from simulation.lsystems1.three.fractal import turtle_steer as steering


class Point:
    """ An x-y-z coordinate"""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def loc(self):
        return self.x, self.y, self.z

    def __eq__(self, other):
        if isinstance(other, Point):
            return round(self.x, 5) == round(other.x, 5) and \
                round(self.y, 5) == round(other.y, 5) and \
                round(self.z, 5) == round(other.z, 5)
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __str__(self):
        return f"Point({round(self.x, 5)}, {round(self.y, 5)}, {round(self.z, 5)})"

    def distance_to(self, other):
        """Euclidean distance b/w 2 points"""
        if isinstance(other, Point):
            dx = self.x - other.x
            dy = self.y - other.y
            dz = self.z - other.z
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            return distance
        raise TypeError("Distance can only be calculated between two Point instances.")


class TurtleLine:
    """In 2D, a state is the x, y of the tip and the angle/orientation of this line segment."""

    def __init__(self, start: Point, hlu, width):
        self.start = start
        self.end = None  # end of the line-segment is added in-flight.
        self.hlu = hlu  # Heading left up vector.

        self.width = width  # default

    def __str__(self):
        return f"""{self.__class__.__name__}: (start={self.start}, end={self.end}, width={round(self.width, 3)}"""

    def is_nan(self):
        if self.end is None or self.start is None:
            return True

        if math.isnan(self.start.x) or math.isnan(self.start.y) or math.isnan(self.start.z):
            return True

        if math.isnan(self.end.x) or math.isnan(self.end.y) or math.isnan(self.end.z):
            return True

        return False

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start.x, self.start.y, self.start.z, self.end.x, self.end.y, self.end.z))


class TurtleBranch(NodeMixin):
    """ An intermediate class with the anytree hierarchy between Turtle Line and Tree Branch
        TurtleLine => TurtleBranch => TreeBranch """

    def __init__(self, turtle_line: TurtleLine, parent=None):
        super().__init__()

        # Anytree parent in the hierarch. this will be preserved for TreeBranch as well.
        self.parent = parent

        # Turtle related configs currently it is a 1-1 mapping
        self.turtle_line = turtle_line

    def t_name(self):
        return str(self.turtle_line).replace('[', '').replace(']', '').replace("'", '') \
            .replace(self.turtle_line.__class__.__name__, self.__class__.__name__) \
            .replace(self.__class__.__name__, '').replace('Point', '').replace(':', '')

    def __eq__(self, other):
        return self.turtle_line == other.turtle_line

    def __hash__(self):
        return hash(self.turtle_line)


def translate(point, length, heading):
    """
    Translate a point xyz by the given length along the heading vector.

    Parameters:
    - Point: Coordinates before translation.
    - length (float): Length of translation.
    - heading (list): Translation heading vector.

    Returns:
    - Translated Point coordinates.
    """
    x = point.x + length * heading[0]
    y = point.y + length * heading[1]
    z = point.z + length * heading[2]
    return Point(x, y, z)


def l_string_to_turtle_lines(l_string, l_config):
    """Take an expanded l_string, l_config to create the TurtleLine collection for drawing/conversion to urdf"""

    def _to_radians(angle):
        return angle * math.pi / 180

    def _parse_angle(_command):
        turn_angle = parser.extract_values(token=_command)
        return _to_radians(turn_angle[0])

    stack = []

    theta_r = _to_radians(l_config.theta)
    sigma_r = _to_radians(l_config.sigma)

    init_hlu = ([0, 0, 1],
                [-math.sin(theta_r), -math.cos(theta_r), 0],
                [-math.cos(theta_r), math.sin(theta_r), 0])

    t_line = TurtleLine(start=Point(0, 0, 0), hlu=init_hlu, width=1.)
    l_tokens = parser.tokenize(l_string)

    t_lines = []  # Lines to draw
    tropism = [t * l_config.e for t in l_config.T]

    for command in l_tokens:
        start, hlu, line_width = t_line.start, t_line.hlu, t_line.width

        if command.startswith('F'):  # Move forward
            seg_lens = parser.extract_values(token=command)
            t_line.end = translate(point=start, length=seg_lens[0], heading=hlu[0])
            # applying tropism is optional, but gives you the nice arch on the branches from gravity.
            t_line.hlu = steering.apply_tropism(hlu, tropism=tropism)

        # Page 46, section 1.10.3 Turtle interpretation of parametric words
        elif '(' in command and parser.extract_operand(succ=command) == "+({})":
            alpha = _parse_angle(_command=command)
            hlu = steering.rotate_around_u(random.gauss(alpha, sigma_r), hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif '(' in command and parser.extract_operand(succ=command) == "&({})":
            gamma = _parse_angle(_command=command)
            hlu = steering.rotate_around_l(random.gauss(gamma, sigma_r), hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif '(' in command and parser.extract_operand(succ=command) == "/({})":
            phi = _parse_angle(_command=command)
            hlu = steering.rotate_around_h(random.gauss(phi, sigma_r), hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif command == '[':  # Remember current t_line
            # beginning of a new branch, so set current end as the start
            point = t_line.end if t_line.end is not None else t_line.start
            split_t_line = TurtleLine(start=point, hlu=hlu, width=line_width)
            stack.append(split_t_line)
            t_lines.append(t_line)
            t_line = split_t_line

        elif command == ']':  # Return to previous t_line
            t_line = stack.pop()

        elif '(' in command and parser.extract_operand(succ=command) == "N({})":
            line_width = parser.extract_values(token=command)[0]
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        elif command == '$':
            hlu = steering.keep_l_horizontal(hlu)
            t_line = TurtleLine(start=start, hlu=hlu, width=line_width)

        # Note: We silently ignore unknown commands, such as G, X
    return t_lines


def turtle_lines_to_branches(turtle_lines: List[TurtleLine]):
    """ TurtleLine(s) => TurtleBranch(s)"""

    t_root = TurtleBranch(turtle_line=turtle_lines[0])
    branch_store_by_line = {turtle_lines[0]: t_root}
    branch_store_by_end = {turtle_lines[0].end: t_root}

    for i in range(1, len(turtle_lines)):
        if turtle_lines[i].is_nan():  # skip nan rows if any
            continue
        if turtle_lines[i] not in branch_store_by_line:
            parent = branch_store_by_end[turtle_lines[i].start]
            branch = TurtleBranch(turtle_line=turtle_lines[i], parent=parent)
            branch_store_by_line[turtle_lines[i]] = branch
            branch_store_by_end[turtle_lines[i].end] = branch

    return t_root
