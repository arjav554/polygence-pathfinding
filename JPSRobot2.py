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

# Create a 30x30 grid of EMPTY cells (matching your explicit dimensions)
grid = arjav.full((30, 30), EMPTY, dtype=int)

# Choose start and goal locations (row, col)
start = (0, 0)
goal  = (29, 29)

grid[start] = START
grid[goal]  = GOAL

# Add a few walls by hand 
# Random Walls (Seeded for consistency)
import random
random.seed(38)
for r in range(30):
    for c in range(30):
        if (r,c) == start or (r,c) == goal:
            continue
        if random.random() < 0.40:
            grid[r, c] = WALL

def print_grid(grid):
    symbols = {
        EMPTY: ".",
        WALL:  "#",
        START: "S",
        GOAL:  "G",
    }
    for r in range(grid.shape[0]):
        row_vals = grid[r]
        line = "".join(symbols[val] for val in row_vals)
        print(line)

print_grid(grid)


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
        # Euclidean distance often works slightly better for JPS/Diagonal, 
        # but Manhattan is acceptable. Using Euclidean here for smoother diagonal preference.
        r, c = cell
        gr, gc = self.goal
        return ((r - gr)**2 + (c - gc)**2)**0.5

    # JPS SPECIFIC: Get pruned neighbors based on parent direction
    def get_jps_successors(self, current, parent):
        successors = []
        r, c = current
        
        # Directions
        if parent:
            pr, pc = parent
            dx = int(arjav.sign(r - pr))
            dy = int(arjav.sign(c - pc))
            
            # If diagonal move
            if dx != 0 and dy != 0:
                # Natural neighbors
                if not self.is_blocked((r + dx, c + dy)):
                    successors.append((r + dx, c + dy))
                if not self.is_blocked((r + dx, c)):
                    successors.append((r + dx, c))
                if not self.is_blocked((r, c + dy)):
                    successors.append((r, c + dy))
                
                # Forced neighbors
                if self.is_blocked((r - dx, c)) and not self.is_blocked((r - dx, c + dy)):
                    successors.append((r - dx, c + dy))
                if self.is_blocked((r, c - dy)) and not self.is_blocked((r + dx, c - dy)):
                    successors.append((r + dx, c - dy))
            
            # If horizontal move
            elif dx == 0:
                if not self.is_blocked((r, c + dy)):
                    successors.append((r, c + dy))
                if self.is_blocked((r + 1, c)) and not self.is_blocked((r + 1, c + dy)):
                    successors.append((r + 1, c + dy))
                if self.is_blocked((r - 1, c)) and not self.is_blocked((r - 1, c + dy)):
                    successors.append((r - 1, c + dy))

            # If vertical move
            else: # dy == 0
                if not self.is_blocked((r + dx, c)):
                    successors.append((r + dx, c))
                if self.is_blocked((r, c + 1)) and not self.is_blocked((r + dx, c + 1)):
                    successors.append((r + dx, c + 1))
                if self.is_blocked((r, c - 1)) and not self.is_blocked((r + dx, c - 1)):
                    successors.append((r + dx, c - 1))
        
        else:
            # If no parent (start node), return all valid 8 neighbors
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if not self.is_blocked((nr, nc)):
                    successors.append((nr, nc))
                    
        return successors

    # JPS SPECIFIC: Scan function to find jump points
    def jump(self, current, direction):
        r, c = current
        dr, dc = direction
        nr, nc = r + dr, c + dc
        
        if self.is_blocked((nr, nc)):
            return None
        
        if (nr, nc) == self.goal:
            return (nr, nc)
        
        # Check for forced neighbors
        # Diagonal Case
        if dr != 0 and dc != 0:
            if (self.is_blocked((r - dr, c + dc)) and not self.is_blocked((r - dr, c + 2*dc))) or \
               (self.is_blocked((r + dr, c - dc)) and not self.is_blocked((r + 2*dr, c - dc))):
                return (nr, nc)
            # Recurse on horizontal/vertical
            if self.jump((nr, nc), (dr, 0)) is not None or \
               self.jump((nr, nc), (0, dc)) is not None:
                return (nr, nc)
                
        # Straight Case
        elif dr != 0: # Vertical
            if (self.is_blocked((nr, nc - 1)) and not self.is_blocked((nr + dr, nc - 1))) or \
               (self.is_blocked((nr, nc + 1)) and not self.is_blocked((nr + dr, nc + 1))):
                return (nr, nc)
        else: # Horizontal
            if (self.is_blocked((nr - 1, nc)) and not self.is_blocked((nr - 1, nc + dc))) or \
               (self.is_blocked((nr + 1, nc)) and not self.is_blocked((nr + 1, nc + dc))):
                return (nr, nc)

        # Recursive call to keep jumping
        return self.jump((nr, nc), direction)


def jps_steps(world: GridWorld):
    """
    Generator for Jump Point Search algorithm steps.
    """
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

        # Identify successors (pruning)
        successors = world.get_jps_successors(current, came_from.get(current))
        
        for neighbor in successors:
            # Calculate direction
            dx = neighbor[0] - current[0]
            dy = neighbor[1] - current[1]
            direction = (dx, dy)
            
            # JUMP!
            jump_point = world.jump(current, direction)
            
            if jump_point:
                # Add jump point to visited for visualization purposes
                visited.add(jump_point)
                
                # dist between current and jump_point
                dist = ((jump_point[0] - current[0])**2 + (jump_point[1] - current[1])**2)**0.5
                tentative_g = g + dist
                
                if jump_point not in g_score or tentative_g < g_score[jump_point]:
                    came_from[jump_point] = current
                    g_score[jump_point] = tentative_g
                    f_score = tentative_g + world.heuristic(jump_point)
                    heapq.heappush(open_set, (f_score, tentative_g, jump_point))


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
            pygame.draw.rect(screen, BLACK, rect, 1)  # cell border

def get_line_points(start, end):
    """Bresenham's like line generation or simple interpolation to fill gaps."""
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
        # Get all points between parent and current (gap filling)
        segment = get_line_points(parent, current)
        # segment includes parent and current. We add them to full_path.
        # reverse segment so we append valid order if needed, but we reverse at end anyway.
        for p in reversed(segment):
             if not full_path or full_path[-1] != p:
                full_path.append(p)
        current = parent
        
    # full_path includes start now
    return full_path[::-1] # Reverse to get Start -> Goal

def run_timing_simulation(world: GridWorld):
    # This runs the generator silently to measure CPU time only
    jps_generator = jps_steps(world)
    path = []
    try:
        for current, came_from, visited in jps_generator:
            if current == world.goal:
                path = reconstruct_path_interpolated(came_from, world.start, world.goal)
                break
    except StopIteration:
        pass
    return path

world = GridWorld(grid=grid, start=start, goal=goal)

# --- Timing Section ---
print("Running JPS Timing Simulation...")
start_time = time.time()
path = run_timing_simulation(world)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"JPS Elapsed time: {elapsed_time:.5e} seconds")
print(f"Path length: {len(path)} steps")
# ----------------------

def run_pygame(world: GridWorld):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("JPS Pathfinding Visualization")

    clock = pygame.time.Clock()
    jps_generator = jps_steps(world)
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
                # Step the JPS Algorithm
                current, came_from, visited = next(jps_generator)
                
                # Check if we hit the goal
                if current == world.goal:
                    path_found = True
                    final_path = reconstruct_path_interpolated(came_from, world.start, world.goal)
                    
            except StopIteration:
                path_found = True # Stopped without finding or finished
                if world.goal in came_from:
                    final_path = reconstruct_path_interpolated(came_from, world.start, world.goal)

        screen.fill(WHITE)
        
        # We pass a set of the path for O(1) lookups in drawing
        path_set = set(final_path) if path_found else set()
        
        draw_world(screen, world, visited, path_set, current)
        pygame.display.flip()
        
        # Speed logic: JPS has fewer steps than A*, so we slow it down slightly 
        # to see the jumps, or speed it up to see the result instantly.
        clock.tick(20) 

run_pygame(world)