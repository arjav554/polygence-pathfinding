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
start = (29, 10)
goal  = (3, 15)
grid[start] = START
grid[goal]  = GOAL

# Walls
walls = [
    (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 14), (15, 15), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (16, 14), (16, 15), (16, 16), (16, 17), (16, 18), (16, 19), (16, 20), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17), (17, 18), (17, 19), (17, 20), (18, 9), (18, 10), (18, 11), (18, 12), (18, 13), (18, 14), (18, 15), (18, 16), (18, 17), (18, 18), (18, 19), (18, 20), (19, 9), (19, 10), (19, 11), (19, 12), (19, 13), (19, 14), (19, 15), (19, 16), (19, 17), (19, 18), (19, 19), (19, 20), (20, 9), (20, 10), (20, 11), (20, 12), (20, 13), (20, 14), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19), (20, 20),
    (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (6, 22), (6, 23), (7, 22), 
    (7, 23), (12, 12), (12, 13), (13, 13), (14, 13), (14, 14), (15, 15), 
    (15, 16), (16, 15), (16, 16), (22, 6), (22, 7), (23, 6), (23, 7), 
    (24, 25), (24, 26), (25, 26), (25, 27), (26, 27),(0, 5), (0, 10), 
    (0, 15), (0, 20), (0, 25), (0, 29), (1, 3), (1, 8), (1, 13), (1, 18), (1, 23), (1, 28), (2, 6), (2, 11), (2, 16), (2, 21), (2, 26), (3, 9), (3, 14), (3, 19), (3, 24), (3, 29), (4, 1), (4, 7), (4, 12), (4, 17), (4, 22), (4, 27), (5, 0), (5, 10), (5, 15), (5, 20), (5, 25), (6, 3), (6, 8), (6, 13), (6, 18), (6, 28), (7, 1), (7, 6), (7, 11), (7, 16), (7, 26), (8, 4), (8, 9), (8, 14), (8, 19), (8, 24), (8, 29), (9, 2), (9, 7), (9, 12), (9, 17), (9, 22), (9, 27), (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (11, 3), (11, 8), (11, 13), (11, 18), (11, 23), (11, 28), (12, 1), (12, 6), (12, 16), (12, 21), (12, 26), (13, 4), (13, 9), (13, 19), (13, 24), (13, 29), (14, 2), (14, 7), (14, 17), (14, 22), (14, 27), (15, 0), (15, 5), (15, 10), (15, 20), (15, 25), (16, 3), (16, 8), (16, 13), (16, 23), (16, 28), (17, 1), (17, 6), (17, 11), (17, 16), (17, 21), (17, 26), (18, 4), (18, 9), (18, 14), (18, 19), (18, 24), (18, 29), (19, 2), (19, 7), (19, 12), (19, 17), (19, 22), (19, 27), (20, 0), (20, 5), (20, 10), (20, 15), (20, 20), (20, 25), (21, 3), (21, 13), (21, 18), (21, 23), (21, 28), (22, 1), (22, 11), (22, 16), (22, 21), (22, 26), (23, 4), (23, 9), (23, 14), (23, 19), (23, 24), (23, 29), (24, 2), (24, 7), (24, 12), (24, 17), (24, 22), (25, 0), (25, 5), (25, 10), (25, 15), (25, 20), (26, 3), (26, 8), (26, 13), (26, 18), (26, 23), (26, 28), (27, 1), (27, 6), (27, 11), (27, 16), (27, 21), (27, 26), (28, 4), (28, 9), (28, 14), (28, 19), (28, 24), (28, 29), (29, 2), (29, 7), (29, 12), (29, 17), (29, 22), (29, 27)
]
for r, c in walls:
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