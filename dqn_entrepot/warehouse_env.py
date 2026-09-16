"""Environnement Gymnasium minimaliste pour un robot d'entrepot."""

import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=10, n_items=3, max_steps=150, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_items = n_items
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.drop_zone = (grid_size - 1, grid_size - 1)
        self.action_space = spaces.Discrete(6)
        low = np.array([0, 0, 0] + [-1, -1] * n_items, dtype=np.float32)
        high = np.array([grid_size - 1, grid_size - 1, 1] + [grid_size - 1, grid_size - 1] * n_items, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.obstacles = set()
        self.items = []

    def _sample_free_position(self, occupied):
        while True:
            position = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
            position = (int(position[0]), int(position[1]))
            if position not in occupied:
                return position

    def _state(self):
        data = [self.robot_pos[0], self.robot_pos[1], int(self.carrying)]
        data += [coordinate for item in self.items for coordinate in item]
        data += [-1, -1] * (self.n_items - len(self.items))
        return np.asarray(data, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = (0, 0)
        self.carrying = False
        self.steps = 0
        self.delivered = 0
        occupied = {self.robot_pos, self.drop_zone}
        self.obstacles = set()
        while len(self.obstacles) < 8:
            obstacle = self._sample_free_position(occupied | self.obstacles)
            self.obstacles.add(obstacle)
        self.items = []
        while len(self.items) < self.n_items:
            item = self._sample_free_position(occupied | self.obstacles | set(self.items))
            self.items.append(item)
        return self._state(), {"delivered": self.delivered}

    def step(self, action):
        self.steps += 1
        reward = -0.2
        row, column = self.robot_pos
        candidate = self.robot_pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        if action in moves:
            delta_row, delta_column = moves[action]
            candidate = (row + delta_row, column + delta_column)
            if not (0 <= candidate[0] < self.grid_size and 0 <= candidate[1] < self.grid_size):
                reward = -2.0
                candidate = self.robot_pos
            elif candidate in self.obstacles:
                reward = -5.0
                candidate = self.robot_pos
            self.robot_pos = candidate
        elif action == 4:  # saisir
            if not self.carrying and self.robot_pos in self.items:
                self.items.remove(self.robot_pos)
                self.carrying = True
                reward = 10.0
            else:
                reward = -1.0
        elif action == 5:  # deposer
            if self.carrying and self.robot_pos == self.drop_zone:
                self.carrying = False
                self.delivered += 1
                reward = 20.0
            else:
                reward = -1.0

        terminated = len(self.items) == 0 and not self.carrying
        if terminated:
            reward += 30.0
        truncated = self.steps >= self.max_steps
        info = {"delivered": self.delivered, "steps": self.steps}
        return self._state(), reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for row, column in self.obstacles:
            grid[row][column] = "X"
        for row, column in self.items:
            grid[row][column] = "I"
        dr, dc = self.drop_zone
        rr, rc = self.robot_pos
        grid[dr][dc] = "D"
        grid[rr][rc] = "R" if not self.carrying else "C"
        return "\n".join(" ".join(line) for line in grid)
