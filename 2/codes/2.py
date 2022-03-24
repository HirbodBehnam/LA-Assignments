from enum import Enum
from functools import cmp_to_key


class Orientation(Enum):
    COLLINEAR = 0
    CLOCK_WISE = 1
    COUNTER_CLOCK_WISE = 2


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"[{self.x:.2f}, {self.y:.2f}]"

    def dist2(self, b) -> float:
        return (self.x - b.x) ** 2 + (self.y - b.y) ** 2


def simple_point_cmp(p1: Point, p2: Point) -> float:
    diff = p1.x - p2.x
    if diff == 0:
        return p1.y - p2.y
    return diff


def orientation(p: Point, q: Point, r: Point) -> Orientation:
    val = ((q.y - p.y) * (r.x - q.x) -
           (q.x - p.x) * (r.y - q.y))
    if val == 0:
        return Orientation.COLLINEAR
    elif val > 0:
        return Orientation.CLOCK_WISE
    else:
        return Orientation.COUNTER_CLOCK_WISE


class Country:
    cities: list[Point]

    def __init__(self):
        self.cities = []
        self.mid_point = Point(0, 0)

    def find_edge_cities_and_find_mid(self) -> list[Point]:
        cities = self.cities.copy()
        # At first find the left most point
        p0 = cities[0]
        p0_index = 0
        for (i, p) in enumerate(cities):
            if (p0.y > p.y) or (p0.y == p.y and p0.x > p.x):
                p0 = p
                p0_index = i
        # Now we have to remove the p0 from list
        del cities[p0_index]

        # From https://www.geeksforgeeks.org/convex-hull-set-2-graham-scan/
        def compare(p1: Point, p2: Point) -> int:
            o = orientation(p0, p1, p2)
            if o == Orientation.COLLINEAR:
                if p0.dist2(p2) >= p0.dist2(p1):
                    return -1
                else:
                    return 1
            else:
                if o == Orientation.COUNTER_CLOCK_WISE:
                    return -1
                else:
                    return 1

        cities.sort(key=cmp_to_key(compare))

        # Remove same elements
        i = -1
        while i < len(cities) - 2:
            i += 1
            if orientation(p0, cities[i], cities[i + 1]) == Orientation.COLLINEAR:
                del cities[i]
                i -= 1
        # Now create our list and push stuff to it
        stack = [p0, cities[0], cities[1]]
        for p in cities[2:]:
            while len(stack) > 1 and orientation(stack[-2], stack[-1], p) != Orientation.COUNTER_CLOCK_WISE:
                stack.pop()
            stack.append(p)

        # Now we have the hull in stack!
        x = 0
        y = 0
        for p in stack:
            x += p.x
            y += p.y
        x /= len(stack)
        y /= len(stack)
        self.mid_point = Point(x, y)
        return stack

    def dist2(self, p: Point) -> float:
        return self.mid_point.dist2(p)


# Read info
line = list(map(int, input().split()))
lines = line[0]
countries = []
for _ in range(line[1]):
    countries.append(Country())
# Read country cities
for _ in range(lines):
    line = input().split()
    c = countries[int(line[0])]
    c.cities.append(Point(float(line[1]), float(line[2])))
for c in countries:  # Find the mid-point
    c.find_edge_cities_and_find_mid()
# Now process all points
for _ in range(int(input())):
    line = list(map(float, input().split()))
    point = Point(line[0], line[1])
    territory_index = 0
    best_dist = countries[0].dist2(point)
    for i in range(1, len(countries)):
        dist = countries[i].dist2(point)
        if best_dist > dist:
            best_dist = dist
            territory_index = i
    print(territory_index, end=' ')
    countries[territory_index].cities.append(point)
print()
# Print cities
for (i, c) in enumerate(countries):
    print(i, ' '.join(map(str, sorted(c.find_edge_cities_and_find_mid(), key=cmp_to_key(simple_point_cmp)))))
