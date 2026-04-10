"""Maze environment implementing the OpenEnv interface.

Generates random solvable mazes using DFS (recursive backtracker)
with extra wall removals to create multiple paths and loops.
"""

import random
from collections import deque
from typing import Optional
from uuid import uuid4

from openenv.core.env_server import Environment

from maze_env.models import MazeAction, MazeObservation, MazeState


class MazeEnvironment(Environment[MazeAction, MazeObservation, MazeState]):
    """12x12 maze navigation with DFS-generated mazes and loops."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    ROWS = 12
    COLS = 12
    MAX_STEPS = 200

    def __init__(self):
        super().__init__()
        self._grid: list[list[str]] = []
        self._agent_pos = (1, 1)
        self._exit_pos = (10, 10)
        self._done = False
        self._step_count = 0
        self._episode_id = ""
        self._maze_seed = 0
        self._optimal_path_length = 0
        self._visited: set[tuple[int, int]] = set()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> MazeObservation:
        self._maze_seed = seed if seed is not None else random.randint(0, 2**31)
        self._episode_id = episode_id or str(uuid4())
        self._done = False
        self._step_count = 0

        rng = random.Random(self._maze_seed)
        self._grid = self._generate_maze(rng)
        self._agent_pos = (1, 1)

        # Place exit at a far corner (bottom-right area)
        exit_r = self.ROWS - 2 if (self.ROWS - 2) % 2 == 1 else self.ROWS - 3
        exit_c = self.COLS - 2 if (self.COLS - 2) % 2 == 1 else self.COLS - 3
        self._exit_pos = (exit_r, exit_c)
        self._grid[exit_r][exit_c] = "."

        self._visited = {self._agent_pos}
        self._optimal_path_length = self._bfs_shortest_path()

        return self._make_observation(reward=0.0)

    def step(
        self,
        action: MazeAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> MazeObservation:
        if self._done:
            return self._make_observation(reward=0.0)

        direction = action.direction.upper()
        dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(
            direction, (0, 0)
        )

        new_r = self._agent_pos[0] + dr
        new_c = self._agent_pos[1] + dc

        # Wall collision
        if (
            new_r < 0
            or new_r >= self.ROWS
            or new_c < 0
            or new_c >= self.COLS
            or self._grid[new_r][new_c] == "#"
        ):
            self._step_count += 1
            reward = -0.3
            if self._step_count >= self.MAX_STEPS:
                self._done = True
                reward = -1.0
            return self._make_observation(reward=reward)

        # Valid move
        old_dist = self._manhattan_to_exit(self._agent_pos)
        self._agent_pos = (new_r, new_c)
        new_dist = self._manhattan_to_exit(self._agent_pos)
        self._step_count += 1

        # Check exit
        if self._agent_pos == self._exit_pos:
            self._done = True
            return self._make_observation(reward=10.0)

        # Reward shaping
        reward = 0.0
        if new_dist < old_dist:
            reward += 0.1
        elif new_dist > old_dist:
            reward -= 0.1

        if self._agent_pos in self._visited:
            reward -= 0.2
        self._visited.add(self._agent_pos)

        if self._step_count >= self.MAX_STEPS:
            self._done = True
            reward = -1.0

        return self._make_observation(reward=reward)

    @property
    def state(self) -> MazeState:
        return MazeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            maze_seed=self._maze_seed,
            optimal_path_length=self._optimal_path_length,
        )

    # ------------------------------------------------------------------
    # Maze generation (DFS + loop creation)
    # ------------------------------------------------------------------

    def _generate_maze(self, rng: random.Random) -> list[list[str]]:
        """Generate a maze using randomized DFS, then remove extra walls
        to create loops and multiple paths."""
        grid = [["#"] * self.COLS for _ in range(self.ROWS)]

        def carve(r: int, c: int):
            grid[r][c] = "."
            dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            rng.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 < nr < self.ROWS - 1 and 0 < nc < self.COLS - 1 and grid[nr][nc] == "#":
                    grid[r + dr // 2][c + dc // 2] = "."
                    carve(nr, nc)

        carve(1, 1)

        # Remove ~15% of interior walls to create loops and alternate paths
        interior_walls = []
        for r in range(2, self.ROWS - 2):
            for c in range(2, self.COLS - 2):
                if grid[r][c] == "#":
                    # Only remove walls that connect two open areas
                    open_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.ROWS and 0 <= nc < self.COLS and grid[nr][nc] == ".":
                            open_neighbors += 1
                    if open_neighbors >= 2:
                        interior_walls.append((r, c))

        rng.shuffle(interior_walls)
        num_to_remove = max(len(interior_walls) // 6, 3)
        for r, c in interior_walls[:num_to_remove]:
            grid[r][c] = "."

        return grid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _manhattan_to_exit(self, pos: tuple[int, int]) -> int:
        return abs(pos[0] - self._exit_pos[0]) + abs(pos[1] - self._exit_pos[1])

    def _bfs_shortest_path(self) -> int:
        """BFS from agent to exit. Returns path length (0 if unreachable)."""
        queue = deque([(self._agent_pos, 0)])
        seen = {self._agent_pos}
        while queue:
            (r, c), dist = queue.popleft()
            if (r, c) == self._exit_pos:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.ROWS
                    and 0 <= nc < self.COLS
                    and self._grid[nr][nc] != "#"
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        return 0

    def _build_display_grid(self) -> list[list[str]]:
        """Build grid with agent (A) and exit (E) markers."""
        grid = [row[:] for row in self._grid]
        er, ec = self._exit_pos
        grid[er][ec] = "E"
        ar, ac = self._agent_pos
        grid[ar][ac] = "A"
        return grid

    def _make_observation(self, reward: float) -> MazeObservation:
        return MazeObservation(
            grid=self._build_display_grid(),
            agent_pos=self._agent_pos,
            exit_pos=self._exit_pos,
            steps_taken=self._step_count,
            done=self._done,
            reward=reward,
        )
