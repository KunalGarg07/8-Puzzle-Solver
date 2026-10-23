"""
8-PUZZLE SOLVER USING A* SEARCH
Course : Artificial Intelligence and Soft Computing (CSE3121)
Topic  : Heuristic Search Techniques (A* Algorithm)

Problem:
    A 3x3 board has tiles 1-8 and one blank (0). A move slides the blank
    up/down/left/right. Reach the goal configuration in the FEWEST moves.

Algorithm:
    A* search with   f(n) = g(n) + h(n)
        g(n) = number of moves made so far (cost from start)
        h(n) = heuristic estimate of moves remaining
    Two admissible heuristics are compared:
        h1 = Misplaced Tiles
        h2 = Manhattan Distance   (stronger / better informed)
    BFS is also run as an uninformed baseline.

Runs as-is in any online Python compiler (no input needed, no libraries).
"""

import heapq
from collections import deque

# ----------------------------------------------------------------------
# 1. PROBLEM DEFINITION
# ----------------------------------------------------------------------
GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)          # 0 represents the blank


def print_board(state):
    """Pretty-print a state as a 3x3 grid."""
    for r in range(3):
        row = state[r * 3:(r + 1) * 3]
        print(" ".join(str(x) if x != 0 else "_" for x in row))
    print()


def is_solvable(state):
    """
    Exactly half of all 8-puzzle states are unsolvable.
    A state is solvable iff the number of INVERSIONS is even
    (blank is ignored; inversion = a bigger tile appearing before a smaller one).
    """
    tiles = [x for x in state if x != 0]
    inversions = sum(1 for i in range(len(tiles))
                       for j in range(i + 1, len(tiles))
                       if tiles[i] > tiles[j])
    return inversions % 2 == 0


def get_neighbors(state):
    """Successor function: all states reachable by one blank move."""
    neighbors = []
    blank = state.index(0)
    row, col = divmod(blank, 3)
    moves = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
    for name, (dr, dc) in moves.items():
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new = list(state)
            swap = nr * 3 + nc
            new[blank], new[swap] = new[swap], new[blank]
            neighbors.append((name, tuple(new)))
    return neighbors


# ----------------------------------------------------------------------
# 2. HEURISTICS
# ----------------------------------------------------------------------
def misplaced_tiles(state):
    """h1: count tiles (excluding blank) that are not in their goal position."""
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != GOAL[i])


def manhattan_distance(state):
    """h2: sum over tiles of |row diff| + |col diff| from goal position."""
    total = 0
    for i, tile in enumerate(state):
        if tile == 0:
            continue
        goal_idx = GOAL.index(tile)
        total += abs(i // 3 - goal_idx // 3) + abs(i % 3 - goal_idx % 3)
    return total


# ----------------------------------------------------------------------
# 3. A* SEARCH
# ----------------------------------------------------------------------
def a_star(start, heuristic):
    """
    Returns (path_of_(move,state), nodes_expanded).
    OPEN list   = priority queue ordered by f = g + h
    CLOSED set  = states already expanded (avoids repeats)
    """
    counter = 0                                   # tie-breaker for the heap
    open_list = [(heuristic(start), counter, 0, start)]   # (f, tie, g, state)
    came_from = {start: (None, None)}             # state -> (parent, move)
    best_g = {start: 0}
    closed = set()
    expanded = 0

    while open_list:
        f, _, g, current = heapq.heappop(open_list)

        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == GOAL:                       # goal test
            path = []
            while came_from[current][0] is not None:
                parent, move = came_from[current]
                path.append((move, current))
                current = parent
            path.reverse()
            return path, expanded

        for move, nxt in get_neighbors(current):
            new_g = g + 1                         # every move costs 1
            if nxt not in best_g or new_g < best_g[nxt]:
                best_g[nxt] = new_g
                came_from[nxt] = (current, move)
                counter += 1
                heapq.heappush(open_list,
                               (new_g + heuristic(nxt), counter, new_g, nxt))
    return None, expanded


# ----------------------------------------------------------------------
# 4. BFS BASELINE (uninformed search, for comparison)
# ----------------------------------------------------------------------
def bfs(start):
    queue = deque([(start, 0)])
    visited = {start}
    expanded = 0
    while queue:
        state, depth = queue.popleft()
        expanded += 1
        if state == GOAL:
            return depth, expanded
        for _, nxt in get_neighbors(state):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))
    return None, expanded


# ----------------------------------------------------------------------
# 5. MAIN PROGRAM
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Change this to try other puzzles (use 0 for the blank)
    START = (1, 3, 6,
             5, 2, 0,
             4, 7, 8)

    # A harder puzzle for the comparison table
    HARD = (8, 1, 3,
            4, 0, 2,
            7, 6, 5)

    print("=" * 50)
    print("8-PUZZLE SOLVER USING A* SEARCH")
    print("=" * 50)
    print("Start state:")
    print_board(START)
    print("Goal state:")
    print_board(GOAL)

    if not is_solvable(START):
        print("This puzzle is UNSOLVABLE (odd number of inversions).")
    else:
        path, expanded = a_star(START, manhattan_distance)
        print(f"Solved with A* (Manhattan) in {len(path)} moves, "
              f"{expanded} nodes expanded.\n")
        print("Solution, step by step:\n")
        print_board(START)
        for i, (move, state) in enumerate(path, 1):
            print(f"Step {i}: move blank {move}")
            print_board(state)

    # ---------------- Heuristic comparison ----------------
    print("=" * 50)
    print("COMPARISON ON A HARDER PUZZLE")
    print("=" * 50)
    print_board(HARD)
    if is_solvable(HARD):
        p1, n1 = a_star(HARD, misplaced_tiles)
        p2, n2 = a_star(HARD, manhattan_distance)
        d3, n3 = bfs(HARD)
        print(f"{'Method':<28}{'Moves':<8}{'Nodes expanded'}")
        print("-" * 50)
        print(f"{'BFS (no heuristic)':<28}{d3:<8}{n3}")
        print(f"{'A* (Misplaced Tiles)':<28}{len(p1):<8}{n1}")
        print(f"{'A* (Manhattan Distance)':<28}{len(p2):<8}{n2}")
        print("\nAll three find the same optimal length;")
        print("better heuristic => fewer nodes explored.")
