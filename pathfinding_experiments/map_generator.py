import numpy as np
import random
from collections import deque

# Cell types
EMPTY = 0
WALL = 1

def generate_empty_grid(size):
    """Generates an open field grid with no obstacles."""
    return np.zeros((size, size), dtype=int)

def generate_random_obstacle_grid(size, density):
    """
    Generates a grid with obstacles scattered randomly.
    density: float between 0.0 and 1.0
    """
    grid = np.zeros((size, size), dtype=int)
    num_cells = size * size
    num_walls = int(num_cells * density)
    
    # Generate random indices
    indices = np.random.choice(num_cells, num_walls, replace=False)
    
    # Set walls
    rows = indices // size
    cols = indices % size
    grid[rows, cols] = WALL
    
    return grid

def generate_maze_grid(size):
    """
    Generates a maze using Recursive Backtracker algorithm.
    Size should strictly be odd for perfect mazes, but we can adapt.
    We will treat even indices as potential "rooms" and odd as "walls".
    """
    # Initialize full of walls
    grid = np.ones((size, size), dtype=int)
    
    # Starting cell (must be odd coordinates to align with wall grid logic if size is odd)
    # Let's simple start at 1,1
    start_r, start_c = 1, 1
    grid[start_r, start_c] = EMPTY
    
    stack = [(start_r, start_c)]
    
    while stack:
        r, c = stack[-1]
        
        # Directions: Up, Down, Left, Right (step 2)
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr < size - 1 and 1 <= nc < size - 1:
                if grid[nr, nc] == WALL:
                    neighbors.append((dr, dc))
        
        if neighbors:
            dr, dc = random.choice(neighbors)
            nr, nc = r + dr, c + dc
            
            # Carve path (wall between)
            grid[r + dr//2, c + dc//2] = EMPTY
            # Carve neighbor
            grid[nr, nc] = EMPTY
            
            stack.append((nr, nc))
        else:
            stack.pop()
            
    return grid

def is_solvable(grid, start, goal):
    """
    Checks if a path exists from start to goal using BFS.
    """
    rows, cols = grid.shape
    if grid[start] == WALL or grid[goal] == WALL:
        return False
        
    queue = deque([start])
    visited = {start}
    
    while queue:
        r, c = queue.popleft()
        
        if (r, c) == goal:
            return True
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # 4-way check is sufficient for reachability if diagonals allowed
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != WALL and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    
    return False

def get_random_valid_coords(grid):
    """Returns a random coordinate that is not a wall."""
    rows, cols = grid.shape
    while True:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if grid[r, c] == EMPTY:
            return (r, c)
