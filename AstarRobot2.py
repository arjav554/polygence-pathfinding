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

# Grid Setup
# Grid Setup
grid = arjav.full((30, 30), EMPTY, dtype=int)
start = (0, 0)
goal  = (29, 29)
grid[start] = START
grid[goal]  = GOAL

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
    start: tuple   
    goal: tuple    
    
    def in_bounds(self, cell):
        r, c = cell
        rows, cols = self.grid.shape
        return 0 <= r < rows and 0 <= c < cols

    def is_blocked(self, cell):
        r, c = cell
        return self.grid[r, c] == WALL

    def neighbors(self, cell):
        r, c = cell
        # 8-way movement
        steps = [
            (-1, 0), (1, 0), (0, -1), (0, 1),       # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)      # Diagonal
        ]
        result = []
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if self.in_bounds(nxt) and not self.is_blocked(nxt):
                result.append(nxt)
        return result

    def heuristic(self, cell):
        # Euclidean for 8-way
        r, c = cell
        gr, gc = self.goal
        return ((r - gr)**2 + (c - gc)**2)**0.5

def astar_steps(world: GridWorld):
    start = world.start
    goal = world.goal

    open_set = []
    # (f, g, cell)
    heapq.heappush(open_set, (0 + world.heuristic(start), 0, start)) 
    came_from = {}
    g_score = {start: 0}

    visited = set()
    visited.add(start)

    while open_set:
        f, g, current = heapq.heappop(open_set)
        
        yield current, came_from, visited

        if current == goal:
            return

        for neighbor in world.neighbors(current):
            # Cost logic: 1.414 for diagonal, 1 for straight
            curr_r, curr_c = current
            neigh_r, neigh_c = neighbor
            
            if curr_r != neigh_r and curr_c != neigh_c:
                step_cost = 1.414
            else:
                step_cost = 1
                
            tentative_g = g + step_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + world.heuristic(neighbor)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                visited.add(neighbor)

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    if current not in came_from and current != start:
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]

def run_timing_simulation(world: GridWorld):
    """
    Runs the A* generator purely for timing, without visualization overhead.
    """
    astar_generator = astar_steps(world)
    path = []
    try:
        for current, came_from, visited in astar_generator:
            if current == world.goal:
                path = reconstruct_path(came_from, world.start, world.goal)
                break
    except StopIteration:
        pass
    return path

# --- Constants & Drawing ---
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

world = GridWorld(grid=grid, start=start, goal=goal)

# --- Timing Section ---
print("Running A* Timing Simulation...")
start_time = time.time()
path = run_timing_simulation(world)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"A* Elapsed time: {elapsed_time:.5e} seconds")
print(f"Path length: {len(path)} steps")
# ----------------------

def run_pygame(world: GridWorld):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("A* Pathfinding Visualization")

    clock = pygame.time.Clock()
    astar_generator = astar_steps(world)
    path_found = False
    came_from = {}
    visited = set()
    current = None
    path = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not path_found:
            try:
                current, came_from, visited = next(astar_generator)
                if current == world.goal:
                     path_found = True
                     path = reconstruct_path(came_from, world.start, world.goal)
            except StopIteration:
                path_found = True
                if world.goal in came_from:
                    path = reconstruct_path(came_from, world.start, world.goal)

        screen.fill(WHITE)
        draw_world(screen, world, visited, set(path) if path_found else set(), current)
        pygame.display.flip()
        clock.tick(20) 

run_pygame(world)