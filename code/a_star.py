from heapq import heappush, heappop
from math import sqrt


def matrix(value):
    """Initialized 200x200 matrix."""
    return [[value] * 200 for _ in range(200)]


def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def heuristic(position, goal, navigation_map):
    h = distance(position, goal)
    adj = adjacency(position, navigation_map)
    h += 9 * (8 - len(adj))
    for n in adj:
        h += (8 - len(adjacency(n, navigation_map)))
    return h


def is_navigable(y, x, navigation_map):
    return x >= 0 and y >= 0 and x < 200 and y < 200 and navigation_map[y, x]


def adjacency(pos, navigation_map):
    adjacent = []
    for i in range(-1, 2):
        y = pos[0] + i
        for j in range(-1, 2):
            x = pos[1] + j
            if (i or j) and is_navigable(y, x, navigation_map):
                adjacent.append((y, x))
    return adjacent


def run(start, goal, navigation_map):
    goals = adjacency(goal, navigation_map)
    closed = set()
    opened = {start}
    parents = matrix(None)
    cost = matrix(1e9)
    cost[start[0]][start[1]] = 0

    score = [(distance(start, goal), start)]  # list of (score, position) tuples

    while opened:
        _, current = heappop(score)
        if current in goals:
            parents[goal[0]][goal[1]] = current
            print("Path {} - {} = {}".format(start, goal, retrace(parents, goal)))
            return retrace(parents, goal)
        if current not in opened:
            continue

        opened.remove(current)
        closed.add(current)

        for n in adjacency(current, navigation_map):
            if n in closed:
                continue

            opened.add(n)

            tentative_cost = cost[current[0]][current[1]] + distance(current, n)
            if tentative_cost < cost[n[0]][n[1]]:
                parents[n[0]][n[1]] = current
                cost[n[0]][n[1]] = tentative_cost
                heappush(score, (tentative_cost + heuristic(n, goal, navigation_map), n))


def retrace(parents, n):
    path = []
    while n:
        path.append(n)
        n = parents[n[0]][n[1]]
    return path[::-1]
