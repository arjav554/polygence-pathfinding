import csv
import sys
import time
import random
import os
from datetime import datetime

# Adjust recursion limit for deep recursion in JPS/Mazes on large grids
sys.setrecursionlimit(5000)

from map_generator import (
    generate_empty_grid, 
    generate_random_obstacle_grid, 
    generate_maze_grid, 
    is_solvable,
    get_random_valid_coords,
    WALL, EMPTY
)
from algorithms import (
    run_astar,
    run_bfs,
    run_dijkstra,
    run_greedy_bfs,
    run_jps
)

# Configuration
GRID_SIZES = [64, 128, 256, 512]
ENV_TYPES = {
    "Open Field": lambda size: generate_empty_grid(size),
    "Random Noise 10%": lambda size: generate_random_obstacle_grid(size, 0.10),
    "Random Noise 20%": lambda size: generate_random_obstacle_grid(size, 0.20),
    "Random Noise 30%": lambda size: generate_random_obstacle_grid(size, 0.30),
    "Random Noise 40%": lambda size: generate_random_obstacle_grid(size, 0.40),
    "Maze": lambda size: generate_maze_grid(size)
}

ALGORITHMS = {
    "A*": run_astar,
    "BFS": run_bfs,
    "Dijkstra": run_dijkstra,
    "Greedy Best-First": run_greedy_bfs,
    "JPS": run_jps
}

TRIALS_PER_CONFIG = 20 # Reduced to 20 for faster execution
OUTPUT_FILE = "pathfinding_research_results.csv"

def run_experiments():
    print(f"Starting Pathfinding Experiments at {datetime.now()}")
    print(f"Results will be saved to: {OUTPUT_FILE}")
    
    # Initialize CSV
    file_exists = os.path.isfile(OUTPUT_FILE)
    mode = 'a' if file_exists else 'w'
    
    with open(OUTPUT_FILE, mode, newline='') as csvfile:
        fieldnames = [
            "Timestamp", "Grid Size", "Environment Type", "Trial ID", 
            "Algorithm", "Nodes Expanded", "Execution Time (ns)", 
            "Path Length", "Peak Memory (Nodes in Open)", "Success"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        total_steps = len(GRID_SIZES) * len(ENV_TYPES) * TRIALS_PER_CONFIG
        current_step = 0
        
        for size in GRID_SIZES:
            for env_name, env_func in ENV_TYPES.items():
                print(f"\n--- Running: Size {size} | Type {env_name} ---")
                
                for trial in range(1, TRIALS_PER_CONFIG + 1):
                    current_step += 1
                    # Progress bar effect
                    if trial % 10 == 0:
                        sys.stdout.write(f"\rTrial {trial}/{TRIALS_PER_CONFIG}")
                        sys.stdout.flush()
                    
                    # 1. Generate Map
                    grid = env_func(size)
                    
                    # 2. Pick Valid Start/Goal
                    # Retry until we find a solvable pair
                    # Limit retries to avoid infinite loops on bad maps (like 100% walls)
                    max_retries = 50
                    start = None
                    goal = None
                    solvable = False
                    
                    for _ in range(max_retries):
                        s = get_random_valid_coords(grid)
                        g = get_random_valid_coords(grid)
                        if s != g and is_solvable(grid, s, g):
                            start = s
                            goal = g
                            solvable = True
                            break
                    
                    if not solvable:
                        print(f"\n[Warning] Could not find solvable path for {env_name} {size}x{size} after {max_retries} attempts. Skipping trial.")
                        continue
                        
                    # 3. Run All Algorithms on EXACT SAME Setup
                    for alg_name, alg_func in ALGORITHMS.items():
                        try:
                            # Run algorithm
                            result = alg_func(grid, start, goal)
                            
                            success = len(result.path) > 0
                            
                            writer.writerow({
                                "Timestamp": datetime.now().isoformat(),
                                "Grid Size": size,
                                "Environment Type": env_name,
                                "Trial ID": trial,
                                "Algorithm": alg_name,
                                "Nodes Expanded": result.nodes_expanded,
                                "Execution Time (ns)": result.execution_time_ns,
                                "Path Length": result.path_length,
                                "Peak Memory (Nodes in Open)": result.peak_memory,
                                "Success": success
                            })
                            
                        except Exception as e:
                            print(f"\n[Error] {alg_name} failed: {e}")
                            writer.writerow({
                                "Timestamp": datetime.now().isoformat(),
                                "Grid Size": size,
                                "Environment Type": env_name,
                                "Trial ID": trial,
                                "Algorithm": alg_name,
                                "Success": False
                            })
                            
                print("") # Newline after progress

    print(f"\nAll experiments complete. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiments()
