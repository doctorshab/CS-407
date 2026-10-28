from collections import deque

def solve_warehouse_navigation():
    # Represent the warehouse as a two-dimensional grid
    warehouse_map = [
        list("#####################"),
        list("#S....#............G#"),
        list("#.##.....##########.#"),
        list("#....##...........#.#"),
        list("#.######.###.#.###..#"),
        list("#.......#.........#.#"),
        list("#####################")
    ]
    
    rows = len(warehouse_map)
    cols = len(warehouse_map[0])
    
    start_pos = None
    goal_pos = None
    
    # Locate Start (S) and Goal (G)
    for r in range(rows):
        for c in range(cols):
            if warehouse_map[r][c] == 'S':
                start_pos = (r, c)
            elif warehouse_map[r][c] == 'G':
                goal_pos = (r, c)
                
    if not start_pos or not goal_pos:
        print("Error: Start or Goal position missing from the map.")
        return

    # BFS Algorithm implementation
    queue = deque([(start_pos[0], start_pos[1], [])])
    visited = set()
    visited.add(start_pos)
    
    # Directions: Up, Down, Left, Right
    directions = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
    
    while queue:
        r, c, path = queue.popleft()
        
        if (r, c) == goal_pos:
            print(f"Goal reached! Path found in {len(path)} steps.")
            print("Move sequence:", " -> ".join(path))
            return path
            
        for dr, dc, move_name in directions:
            nr, nc = r + dr, c + dc
            
            # Check boundaries and obstacles
            if 0 <= nr < rows and 0 <= nc < cols:
                if warehouse_map[nr][nc] != '#' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, path + [move_name]))
                    
    print("No collision-free path exists from S to G.")
    return None

if __name__ == "__main__":
    solve_warehouse_navigation()