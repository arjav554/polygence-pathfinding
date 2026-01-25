import numpy as arjav
import heapq
import pygame
import sys
import time
from dataclasses import dataclass

# Cell types
EMPTY = 0
WALL = 1
START = 2
GOAL = 3 

# Create a 30x30 grid of EMPTY cells
grid = arjav.full((30, 30), EMPTY, dtype=int)

# Choose start and goal locations (row, col)
start = (0, 0)
goal  = (29, 29)

grid[start] = START
grid[goal]  = GOAL

# Add a few walls by hand (keeping your original map)
# Random Walls (Seeded for consistency)
import random
random.seed(38)
for r in range(30):
    for c in range(30):
        if (r,c) == start or (r,c) == goal:
            continue
        if random.random() < 0.40:
            grid[r, c] = WALL

@dataclass
class GridWorld:
    grid: arjav.ndarray
    start: tuple   # (row, col)
    goal: tuple    # (row, col)

    def in_bounds(self, cell):
        r, c = cell
        rows, cols = self.grid.shape
        return 0 <= r < rows and 0 <= c < cols

    def is_blocked(self, cell):
        if not self.in_bounds(cell):
            return True
        r, c = cell
        return self.grid[r, c] == WALL

    def heuristic(self, cell):
        # DIJKSTRA CHANGE: Return 0. Dijkstra is essentially A* with h(x) = 0.
        return 0

    # DIJKSTRA CHANGE: Standard 8-way neighbors instead of JPS pruning
    def get_neighbors(self, current):
        neighbors = []
        r, c = current
        
        # All 8 directions
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if not self.is_blocked((nr, nc)):
                neighbors.append((nr, nc))
        
        return neighbors

    # DIJKSTRA CHANGE: 'jump' method removed entirely.


def dijkstra_steps(world: GridWorld):
    """
    Generator for Dijkstra's algorithm steps.
    """
    start = world.start
    goal = world.goal

    open_set = []
    # (f, g, cell). For Dijkstra f = g (since h=0).
    heapq.heappush(open_set, (0, 0, start)) 
    came_from = {}
    g_score = {start: 0}
    
    visited = set()
    visited.add(start)

    while open_set:
        f, g, current = heapq.heappop(open_set)
        
        yield current, came_from, visited

        if current == goal:
            return

        # DIJKSTRA CHANGE: Iterate standard neighbors instead of jumping
        successors = world.get_neighbors(current)
        
        for neighbor in successors:
            # Calculate distance (1 for straight, sqrt(2) for diagonal)
            dist = ((neighbor[0] - current[0])**2 + (neighbor[1] - current[1])**2)**0.5
            tentative_g = g + dist
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # f_score is just tentative_g for Dijkstra
                heapq.heappush(open_set, (tentative_g, tentative_g, neighbor))
                visited.add(neighbor)


CELL_SIZE = 20
ROWS, COLS = grid.shape
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (100, 100, 100)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)

def draw_world(screen, world: GridWorld, visited=set(), path=set(), current=None):
    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if world.grid[r, c] == WALL:
                color = BLACK
            elif world.grid[r, c] == START:
                color = GREEN
            elif world.grid[r, c] == GOAL:
                color = RED
            elif cell in path:
                color = YELLOW
            elif cell == current:
                color = BLUE
            elif cell in visited:
                color = GRAY
            else:
                color = WHITE
                
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

def get_line_points(start, end):
    points = []
    r0, c0 = start
    r1, c1 = end
    
    steps = max(abs(r1 - r0), abs(c1 - c0))
    if steps == 0:
        return [start]
        
    for i in range(steps + 1):
        r = int(r0 + (r1 - r0) * i / steps)
        c = int(c0 + (c1 - c0) * i / steps)
        points.append((r, c))
    return points

def reconstruct_path_interpolated(came_from, start, goal):
    full_path = []
    current = goal
    
    if current not in came_from and current != start:
        return []

    while current != start:
        parent = came_from[current]
        segment = get_line_points(parent, current)
        for p in reversed(segment):
             if not full_path or full_path[-1] != p:
                full_path.append(p)
        current = parent
        
    return full_path[::-1] 

def run_timing_simulation(world: GridWorld):
    # Updated to call dijkstra_steps
    d_generator = dijkstra_steps(world)
    path = []
    try:
        for current, came_from, visited in d_generator:
            if current == world.goal:
                path = reconstruct_path_interpolated(came_from, world.start, world.goal)
                break
    except StopIteration:
        pass
    return path

world = GridWorld(grid=grid, start=start, goal=goal)

# --- Timing Section ---
print("Running Dijkstra Timing Simulation...")
start_time = time.time()
path = run_timing_simulation(world)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Dijkstra Elapsed time: {elapsed_time:.5e} seconds")
print(f"Path length: {len(path)} steps")
# ----------------------

def run_pygame(world: GridWorld):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dijkstra Pathfinding Visualization")

    clock = pygame.time.Clock()
    # Updated generator call
    d_generator = dijkstra_steps(world)
    path_found = False
    came_from = {}
    visited = set()
    current = None
    final_path = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not path_found:
            try:
                current, came_from, visited = next(d_generator)
                
                if current == world.goal:
                    path_found = True
                    final_path = reconstruct_path_interpolated(came_from, world.start, world.goal)
                    
            except StopIteration:
                path_found = True 
                if world.goal in came_from:
                    final_path = reconstruct_path_interpolated(came_from, world.start, world.goal)

        screen.fill(WHITE)
        
        path_set = set(final_path) if path_found else set()
        
        draw_world(screen, world, visited, path_set, current)
        pygame.display.flip()
        
        # Speeded up slightly as Dijkstra has many more steps than JPS
        clock.tick(20) 

run_pygame(world)