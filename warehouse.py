import heapq
from collections import deque

def parse_warehouse(grid):
    """Finds starting and goal coordinates in the grid."""
    rows, cols = len(grid), len(grid[0])
    start, goal = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
    return start, goal, rows, cols

def manhattan_distance(p1, p2):
    """Heuristic function h(n): Manhattan distance."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# ----------------------------------------------------
# 1. Breadth-First Search (BFS) - Blind Search
# ----------------------------------------------------
def bfs_search(grid):
    start, goal, rows, cols = parse_warehouse(grid)
    if not start or not goal:
        return None

    # Queue stores: (current_state, path)
    queue = deque([(start, [])])
    visited = {start}
    directions = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
    states_expanded = 0

    while queue:
        (r, c), path = queue.popleft()
        states_expanded += 1

        if (r, c) == goal:
            return {"name": "BFS", "found": True, "path": path, "length": len(path), "expanded": states_expanded}

        for dr, dc, action in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [action]))

    return {"name": "BFS", "found": False, "path": None, "length": None, "expanded": states_expanded}

# ----------------------------------------------------
# 2. Depth-First Search (DFS) - Blind Search
# ----------------------------------------------------
def dfs_search(grid):
    start, goal, rows, cols = parse_warehouse(grid)
    if not start or not goal:
        return None

    # Stack stores: (current_state, path)
    stack = [(start, [])]
    visited = set()
    directions = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
    states_expanded = 0

    while stack:
        (r, c), path = stack.pop()

        if (r, c) in visited:
            continue
        visited.add((r, c))
        states_expanded += 1

        if (r, c) == goal:
            return {"name": "DFS", "found": True, "path": path, "length": len(path), "expanded": states_expanded}

        for dr, dc, action in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and (nr, nc) not in visited:
                stack.append(((nr, nc), path + [action]))

    return {"name": "DFS", "found": False, "path": None, "length": None, "expanded": states_expanded}

# ----------------------------------------------------
# 3. Greedy Best-First Search - Informed Search
# ----------------------------------------------------
def greedy_search(grid):
    start, goal, rows, cols = parse_warehouse(grid)
    if not start or not goal:
        return None

    # Priority Queue stores: (h(n), state, path)
    # Evaluates purely by estimated distance to goal: f(n) = h(n)
    entry_id = 0  # Tie-breaker for heap
    frontier = [(manhattan_distance(start, goal), entry_id, start, [])]
    visited = set()
    directions = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
    states_expanded = 0

    while frontier:
        _, _, (r, c), path = heapq.heappop(frontier)

        if (r, c) in visited:
            continue
        visited.add((r, c))
        states_expanded += 1

        if (r, c) == goal:
            return {"name": "Greedy Best-First", "found": True, "path": path, "length": len(path), "expanded": states_expanded}

        for dr, dc, action in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and (nr, nc) not in visited:
                entry_id += 1
                h = manhattan_distance((nr, nc), goal)
                heapq.heappush(frontier, (h, entry_id, (nr, nc), path + [action]))

    return {"name": "Greedy Best-First", "found": False, "path": None, "length": None, "expanded": states_expanded}

# ----------------------------------------------------
# 4. A* Search - Informed Search
# ----------------------------------------------------
def a_star_search(grid):
    start, goal, rows, cols = parse_warehouse(grid)
    if not start or not goal:
        return None

    # Priority Queue stores: (f(n), g(n), entry_id, state, path)
    # Evaluation function: f(n) = g(n) + h(n)
    entry_id = 0
    h_start = manhattan_distance(start, goal)
    frontier = [(h_start, 0, entry_id, start, [])]
    visited = set()
    directions = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
    states_expanded = 0

    while frontier:
        _, g, _, (r, c), path = heapq.heappop(frontier)

        if (r, c) in visited:
            continue
        visited.add((r, c))
        states_expanded += 1

        if (r, c) == goal:
            return {"name": "A*", "found": True, "path": path, "length": len(path), "expanded": states_expanded}

        for dr, dc, action in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and (nr, nc) not in visited:
                entry_id += 1
                new_g = g + 1
                h = manhattan_distance((nr, nc), goal)
                f = new_g + h
                heapq.heappush(frontier, (f, new_g, entry_id, (nr, nc), path + [action]))

    return {"name": "A*", "found": False, "path": None, "length": None, "expanded": states_expanded}