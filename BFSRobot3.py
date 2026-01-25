import numpy as arjav 
import heapq
import pygame
import sys
import time
from collections import deque
from dataclasses import dataclass

# Cell types
EMPTY = 0
WALL = 1
START = 2
GOAL = 3 

# Create grid
grid = arjav.full((30, 30), EMPTY, dtype=int)

# Choose start and goal locations
start = (0, 0)
goal  = (29, 29)

grid[start] = START
grid[goal]  = GOAL

# Add a few walls
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
        r, c = cell
        return self.grid[r, c] == WALL

    def neighbors(self, cell):
        r, c = cell
        # CHANGED: Added diagonals (8-way movement)
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

    def draw(self, path=None):
        # Text-based draw (optional debug)
        display = self.grid.copy()
        if path is not None:
            for cell in path:
                if cell != self.start and cell != self.goal:
                    r, c = cell
                    display[r, c] = 4 
        symbols = {
            EMPTY: ".",
            WALL:  "#",
            START: "S",
            GOAL:  "G",
            4:     "*",
        }
        for r in range(display.shape[0]):
            line = "".join(symbols[val] for val in display[r])
            print(line)

def bfs_steps(world: GridWorld):
    start = world.start
    goal = world.goal

    queue = deque([start])
    came_from = {}
    visited = set([start])

    while queue:
        current = queue.popleft()

        yield current, came_from, visited

        if current == goal:
            return

        for neighbor in world.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    # Safety check: if goal wasn't reached
    if current not in came_from and current != start:
        return []
        
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1] # Reverse to get Start -> Goal

def run_timing_simulation(world: GridWorld):
    # Runs the generator purely for timing
    generator = bfs_steps(world)
    path = []
    try:
        for current, came_from, visited in generator:
            if current == world.goal:
                path = reconstruct_path(came_from, world.start, world.goal)
                break
    except StopIteration:
        pass
    return path

CELL_SIZE = 20
ROWS, COLS = grid.shape
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
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
            elif cell == current:
                color = BLUE
            elif cell in path:
                color = YELLOW
            elif cell in visited:
                color = GRAY
            else:
                color = WHITE
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, LIGHT_GRAY, rect, 1)

world = GridWorld(grid=grid, start=start, goal=goal)

# --- Timing Section ---
print("Running BFS Timing Simulation...")
start_time = time.time()
path = run_timing_simulation(world)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"BFS Elapsed time: {elapsed_time:.5e} seconds")
print(f"Path length: {len(path)} steps")
# ----------------------

def run_pygame(world: GridWorld):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BFS Pathfinding Visualization (8-Way)")

    clock = pygame.time.Clock()
    bfs_generator = bfs_steps(world)
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
                current, came_from, visited = next(bfs_generator)
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