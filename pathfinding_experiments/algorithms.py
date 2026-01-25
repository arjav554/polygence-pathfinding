import heapq
import time
import numpy as np
from dataclasses import dataclass
from collections import deque
import sys

@dataclass
class PathResult:
    path: list
    nodes_expanded: int
    execution_time_ns: int
    path_length: float
    peak_memory: int

# Cell types
EMPTY = 0
WALL = 1

class GridWorld:
    def __init__(self, grid, goal):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.goal = goal

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_blocked(self, r, c):
        if not self.in_bounds(r, c):
            return True
        return self.grid[r, c] == WALL

    def neighbors(self, r, c):
        steps = [
            (-1, 0), (1, 0), (0, -1), (0, 1),       # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)      # Diagonal
        ]
        result = []
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and not self.is_blocked(nr, nc):
                result.append((nr, nc))
        return result

    def heuristic(self, r, c):
        # Euclidean distance
        gr, gc = self.goal
        return ((r - gr)**2 + (c - gc)**2)**0.5

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return []
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    return path[::-1]

def calculate_path_length(path):
    if not path:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        dist = ((r1 - r2)**2 + (c1 - c2)**2)**0.5
        length += dist
    return length

# --- Algorithms ---

def run_astar(grid, start, goal):
    start_time = time.perf_counter_ns()
    nodes_expanded = 0
    peak_memory = 0
    
    world = GridWorld(grid, goal)
    
    open_set = []
    # (f, g, r, c)
    heapq.heappush(open_set, (world.heuristic(*start), 0, start[0], start[1]))
    peak_memory = 1
    
    came_from = {}
    g_score = {start: 0}
    
    final_path = []
    
    while open_set:
        peak_memory = max(peak_memory, len(open_set))
        f, g, r, c = heapq.heappop(open_set)
        current = (r, c)
        
        nodes_expanded += 1
        
        if current == goal:
            final_path = reconstruct_path(came_from, start, goal)
            break
            
        for nr, nc in world.neighbors(r, c):
            neighbor = (nr, nc)
            # Cost: 1 for cardinal, 1.414 for diagonal
            if r != nr and c != nc:
                cost = 1.41421356
            else:
                cost = 1.0
                
            tentative_g = g + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + world.heuristic(nr, nc)
                heapq.heappush(open_set, (f_score, tentative_g, nr, nc))
                
    end_time = time.perf_counter_ns()
    return PathResult(
        path=final_path,
        nodes_expanded=nodes_expanded,
        execution_time_ns=end_time - start_time,
        path_length=calculate_path_length(final_path),
        peak_memory=peak_memory
    )

def run_dijkstra(grid, start, goal):
    start_time = time.perf_counter_ns()
    nodes_expanded = 0
    peak_memory = 0
    
    world = GridWorld(grid, goal)
    
    open_set = []
    # (g, r, c) - Dijkstra is A* with h=0, so f=g
    heapq.heappush(open_set, (0, start[0], start[1]))
    peak_memory = 1
    
    came_from = {}
    g_score = {start: 0}
    
    final_path = []
    
    while open_set:
        peak_memory = max(peak_memory, len(open_set))
        g, r, c = heapq.heappop(open_set)
        current = (r, c)
        
        nodes_expanded += 1
        
        if current == goal:
            final_path = reconstruct_path(came_from, start, goal)
            break
            
        for nr, nc in world.neighbors(r, c):
            neighbor = (nr, nc)
            if r != nr and c != nc:
                cost = 1.41421356
            else:
                cost = 1.0
                
            tentative_g = g + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, nr, nc))
                
    end_time = time.perf_counter_ns()
    return PathResult(
        path=final_path,
        nodes_expanded=nodes_expanded,
        execution_time_ns=end_time - start_time,
        path_length=calculate_path_length(final_path),
        peak_memory=peak_memory
    )

def run_bfs(grid, start, goal):
    start_time = time.perf_counter_ns()
    nodes_expanded = 0
    peak_memory = 0
    
    world = GridWorld(grid, goal)
    
    queue = deque([start])
    visited = {start}
    came_from = {}
    
    peak_memory = 1
    final_path = []
    
    while queue:
        peak_memory = max(peak_memory, len(queue))
        current = queue.popleft()
        nodes_expanded += 1
        
        if current == goal:
            final_path = reconstruct_path(came_from, start, goal)
            break
            
        r, c = current
        for nr, nc in world.neighbors(r, c):
            neighbor = (nr, nc)
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                
    end_time = time.perf_counter_ns()
    return PathResult(
        path=final_path,
        nodes_expanded=nodes_expanded,
        execution_time_ns=end_time - start_time,
        path_length=calculate_path_length(final_path),
        peak_memory=peak_memory
    )

def run_greedy_bfs(grid, start, goal):
    start_time = time.perf_counter_ns()
    nodes_expanded = 0
    peak_memory = 0
    
    world = GridWorld(grid, goal)
    
    open_set = []
    # (h, r, c) - Only heuristic matters
    heapq.heappush(open_set, (world.heuristic(*start), start[0], start[1]))
    peak_memory = 1
    
    came_from = {}
    visited = {start}
    
    final_path = []
    
    while open_set:
        peak_memory = max(peak_memory, len(open_set))
        h, r, c = heapq.heappop(open_set)
        current = (r, c)
        
        nodes_expanded += 1
        
        if current == goal:
            final_path = reconstruct_path(came_from, start, goal)
            break
            
        for nr, nc in world.neighbors(r, c):
            neighbor = (nr, nc)
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                priority = world.heuristic(nr, nc)
                heapq.heappush(open_set, (priority, nr, nc))
                
    end_time = time.perf_counter_ns()
    return PathResult(
        path=final_path,
        nodes_expanded=nodes_expanded,
        execution_time_ns=end_time - start_time,
        path_length=calculate_path_length(final_path),
        peak_memory=peak_memory
    )

# --- JPS Implementation ---

def get_jps_successors(world, r, c, parent):
    successors = []
    if parent is None:
        # Start node: return all valid neighbors
        return world.neighbors(r, c)
    
    pr, pc = parent
    dx = int(np.sign(r - pr))
    dy = int(np.sign(c - pc))
    
    # Diagonal
    if dx != 0 and dy != 0:
        if not world.is_blocked(r + dx, c + dy): successors.append((r + dx, c + dy))
        if not world.is_blocked(r + dx, c): successors.append((r + dx, c))
        if not world.is_blocked(r, c + dy): successors.append((r, c + dy))
        
        # Forced
        if world.is_blocked(r - dx, c) and not world.is_blocked(r - dx, c + dy):
            successors.append((r - dx, c + dy))
        if world.is_blocked(r, c - dy) and not world.is_blocked(r + dx, c - dy):
            successors.append((r + dx, c - dy))
            
    # Horizontal
    elif dx == 0:
        if not world.is_blocked(r, c + dy): successors.append((r, c + dy))
        if world.is_blocked(r + 1, c) and not world.is_blocked(r + 1, c + dy):
            successors.append((r + 1, c + dy))
        if world.is_blocked(r - 1, c) and not world.is_blocked(r - 1, c + dy):
            successors.append((r - 1, c + dy))
            
    # Vertical
    else: # dy == 0
        if not world.is_blocked(r + dx, c): successors.append((r + dx, c))
        if world.is_blocked(r, c + 1) and not world.is_blocked(r + dx, c + 1):
            successors.append((r + dx, c + 1))
        if world.is_blocked(r, c - 1) and not world.is_blocked(r + dx, c - 1):
            successors.append((r + dx, c - 1))
            
    return successors

def jump(world, r, c, dr, dc):
    nr, nc = r + dr, c + dc
    
    if not world.in_bounds(nr, nc) or world.is_blocked(nr, nc):
        return None
        
    if (nr, nc) == world.goal:
        return (nr, nc)
        
    # Forced neighbors check
    if dr != 0 and dc != 0: # Diagonal
        if (world.in_bounds(r - dr, c + dc) and world.is_blocked(r - dr, c + dc) and \
            world.in_bounds(r - dr, c + 2*dc) and not world.is_blocked(r - dr, c + 2*dc)) or \
           (world.in_bounds(r + dr, c - dc) and world.is_blocked(r + dr, c - dc) and \
            world.in_bounds(r + 2*dr, c - dc) and not world.is_blocked(r + 2*dr, c - dc)):
            return (nr, nc)
        # Recurse
        if jump(world, nr, nc, dr, 0) is not None or jump(world, nr, nc, 0, dc) is not None:
             return (nr, nc)
    elif dr != 0: # Vertical
        if (world.in_bounds(nr, nc - 1) and world.is_blocked(nr, nc - 1) and \
            world.in_bounds(nr + dr, nc - 1) and not world.is_blocked(nr + dr, nc - 1)) or \
           (world.in_bounds(nr, nc + 1) and world.is_blocked(nr, nc + 1) and \
            world.in_bounds(nr + dr, nc + 1) and not world.is_blocked(nr + dr, nc + 1)):
            return (nr, nc)
    else: # Horizontal
        if (world.in_bounds(nr - 1, nc) and world.is_blocked(nr - 1, nc) and \
            world.in_bounds(nr - 1, nc + dc) and not world.is_blocked(nr - 1, nc + dc)) or \
           (world.in_bounds(nr + 1, nc) and world.is_blocked(nr + 1, nc) and \
            world.in_bounds(nr + 1, nc + dc) and not world.is_blocked(nr + 1, nc + dc)):
            return (nr, nc)
            
    return jump(world, nr, nc, dr, dc)

def run_jps(grid, start, goal):
    start_time = time.perf_counter_ns()
    nodes_expanded = 0
    peak_memory = 0
    
    world = GridWorld(grid, goal)
    
    open_set = []
    # (f, g, r, c)
    heapq.heappush(open_set, (world.heuristic(*start), 0, start[0], start[1]))
    peak_memory = 1
    
    came_from = {}
    g_score = {start: 0}
    
    final_path = []
    
    while open_set:
        peak_memory = max(peak_memory, len(open_set))
        f, g, r, c = heapq.heappop(open_set)
        current = (r, c)
        
        nodes_expanded += 1
        
        if current == goal:
            final_path = reconstruct_path(came_from, start, goal)
            # Interpolate path for accurate length
            interpolated = []
            if final_path:
                for i in range(len(final_path) - 1):
                    p1 = final_path[i]
                    p2 = final_path[i+1]
                    # Bresenham/Linear fill
                    r0, c0 = p1
                    r1, c1 = p2
                    steps = max(abs(r1-r0), abs(c1-c0))
                    for s in range(steps):
                        inter_r = int(r0 + (r1 - r0) * s / steps)
                        inter_c = int(c0 + (c1 - c0) * s / steps)
                        interpolated.append((inter_r, inter_c))
                interpolated.append(goal)
            final_path = interpolated
            break

        parent = came_from.get(current)
        successors = get_jps_successors(world, r, c, parent)
        
        for nr, nc in successors:
            dx = nr - r
            dy = nc - c
            
            jump_point = jump(world, r, c, dx, dy)
            
            if jump_point:
                jr, jc = jump_point
                dist = ((jr - r)**2 + (jc - c)**2)**0.5
                tentative_g = g + dist
                
                if jump_point not in g_score or tentative_g < g_score[jump_point]:
                    came_from[jump_point] = current
                    g_score[jump_point] = tentative_g
                    f_score = tentative_g + world.heuristic(jr, jc)
                    heapq.heappush(open_set, (f_score, tentative_g, jr, jc))
                    
    end_time = time.perf_counter_ns()
    return PathResult(
        path=final_path,
        nodes_expanded=nodes_expanded,
        execution_time_ns=end_time - start_time,
        path_length=calculate_path_length(final_path),
        peak_memory=peak_memory
    )
