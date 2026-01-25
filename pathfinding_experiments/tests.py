import unittest
from map_generator import generate_empty_grid
from algorithms import (
    run_astar,
    run_bfs,
    run_dijkstra,
    run_greedy_bfs,
    run_jps
)

class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.size = 20
        self.grid = generate_empty_grid(self.size)
        self.start = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def run_alg_test(self, alg_func):
        print(f"\nTesting {alg_func.__name__}...", end="")
        result = alg_func(self.grid, self.start, self.goal)
        print(f" Path Len: {len(result.path)}")
        self.assertGreater(len(result.path), 0, "Path not found")
        self.assertEqual(result.path[0], self.start, "Start mismatch")
        self.assertEqual(result.path[-1], self.goal, "Goal mismatch")

    def test_astar(self):
        self.run_alg_test(run_astar)

    def test_bfs(self):
        self.run_alg_test(run_bfs)

    def test_dijkstra(self):
        self.run_alg_test(run_dijkstra)

    def test_greedy(self):
        self.run_alg_test(run_greedy_bfs)

    def test_jps(self):
        self.run_alg_test(run_jps)

if __name__ == '__main__':
    unittest.main()
