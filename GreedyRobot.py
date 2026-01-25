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
grid = arjav.full((30, 30), EMPTY, dtype=int)
start = (0, 0)
goal  = (29, 29)
grid[start] = START
grid[goal]  = GOAL

# Walls
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

def greedy_bfs_steps(world: GridWorld):
    start = world.start
    goal = world.goal

    open_set = []
    # GREEDY CHANGE: Priority is ONLY the heuristic (h).
    # We store (h, cell) in the heap.
    heapq.heappush(open_set, (world.heuristic(start), start)) 
    came_from = {}
    
    # We use a visited set to avoid loops, similar to BFS
    visited = set()
    visited.add(start)

    while open_set:
        # We only pop based on lowest h
        h, current = heapq.heappop(open_set)
        
        yield current, came_from, visited

        if current == goal:
            return

        for neighbor in world.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                
                # GREEDY CHANGE: Cost (g) is ignored. Priority = heuristic.
                priority = world.heuristic(neighbor)
                heapq.heappush(open_set, (priority, neighbor))

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
    # Updated to call greedy_bfs_steps
    greedy_generator = greedy_bfs_steps(world)
    path = []
    try:
        for current, came_from, visited in greedy_generator:
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
print("Running Greedy BFS Timing Simulation...")
start_time = time.time()
path = run_timing_simulation(world)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Greedy BFS Elapsed time: {elapsed_time:.5e} seconds")
print(f"Path length: {len(path)} steps")
# ----------------------

def run_pygame(world: GridWorld):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Greedy Best-First Search Visualization")

    clock = pygame.time.Clock()
    # Updated generator call
    greedy_generator = greedy_bfs_steps(world)
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
                current, came_from, visited = next(greedy_generator)
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