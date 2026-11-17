import math
import random

from config import (
    GRID_SIZE, MAP_ROWS, MAP_COLS,
    SENSOR_ANGLES, ROTATION_STEP,
    WALL_UP, WALL_DOWN, WALL_LEFT, WALL_RIGHT,
    MIN_RANGE_MM, MAX_RANGE_MM, MIN_RANGE_PX, MAX_RANGE_PX,
    PIXELS_PER_MM, SENSOR_MEASUREMENT_SIGMA_MM,
)
from discrete_noise import (
    forward_transition_probs,
    sample_three_way,
    turn_transition_probs,
)

ACTION_FORWARD = "FORWARD"
ACTION_TURN_LEFT = "TURN_LEFT"
ACTION_TURN_RIGHT = "TURN_RIGHT"

THETA_TO_STEP = {
    0: (0, 1),
    2: (1, 0),
    4: (0, -1),
    6: (-1, 0),
}


class Robot:
    def __init__(self, r, c):
        self.r = r  # grid row
        self.c = c  # grid col
        self.true_x = (c * GRID_SIZE) + (GRID_SIZE / 2)  # pixel position
        self.true_y = (r * GRID_SIZE) + (GRID_SIZE / 2)

        # 0=0°, 1=45°, ..., 7=315°. Start at 2 (90°/South) so W key moves "up".
        self.angle_index = 2

        self.active_sensors = [True] * len(SENSOR_ANGLES)

    @property
    def angle_deg(self):
        return self.angle_index * ROTATION_STEP

    def rotate(self, direction):
        """direction: +1 = clockwise, -1 = counter-clockwise."""
        self.angle_index = (self.angle_index + direction) % 8

    def toggle_sensor(self, idx):
        if 0 <= idx < len(self.active_sensors):
            self.active_sensors[idx] = not self.active_sensors[idx]

    def move_to(self, r, c):
        self.r = r
        self.c = c
        self.true_x = (c * GRID_SIZE) + (GRID_SIZE / 2)
        self.true_y = (r * GRID_SIZE) + (GRID_SIZE / 2)

    def move_rel(self, dr, dc, wall_map):
        """Attempt a one-cell relative move, checking thin walls. Returns True if successful."""
        if dr == -1:
            if wall_map[self.r, self.c, WALL_UP] == 0: return False
            if self.r - 1 < 0: return False
        if dr == 1:
            if wall_map[self.r, self.c, WALL_DOWN] == 0: return False
            if self.r + 1 >= MAP_ROWS: return False
        if dc == -1:
            if wall_map[self.r, self.c, WALL_LEFT] == 0: return False
            if self.c - 1 < 0: return False
        if dc == 1:
            if wall_map[self.r, self.c, WALL_RIGHT] == 0: return False
            if self.c + 1 >= MAP_COLS: return False

        self.move_to(self.r + dr, self.c + dc)
        return True

    def _move_forward_one_cell(self, wall_map):
        step = THETA_TO_STEP.get(self.angle_index % 8)
        if step is None:
            return False
        return self.move_rel(step[0], step[1], wall_map)

    def rotate_noisy(self, direction, rng):
        """Sample a discrete noisy turn outcome and update angle_index."""
        outcome = sample_three_way(turn_transition_probs(noisy=True), rng)
        steps = (0, 1, 2)[outcome]
        self.angle_index = (self.angle_index + direction * steps) % 8
        return steps

    def move_forward_noisy(self, wall_map, rng):
        """Sample a discrete noisy forward outcome and move up to two cells."""
        outcome = sample_three_way(forward_transition_probs(noisy=True), rng)
        steps = (0, 1, 2)[outcome]
        moved = 0
        for _ in range(steps):
            if not self._move_forward_one_cell(wall_map):
                break
            moved += 1
        return moved

    def apply_action(self, action, wall_map, rng, noisy=True):
        """Execute a discrete action on the true simulated robot."""
        if action == ACTION_TURN_LEFT:
            return self.rotate_noisy(-1, rng) if noisy else self.rotate(-1)
        if action == ACTION_TURN_RIGHT:
            return self.rotate_noisy(1, rng) if noisy else self.rotate(1)
        if action == ACTION_FORWARD:
            return self.move_forward_noisy(wall_map, rng) if noisy else self._move_forward_one_cell(wall_map)
        raise ValueError(f"unknown action: {action}")

    def measure(self, wall_map, add_noise=False):
        """
        Ray-cast each sensor and return distances in millimeters.
        add_noise: adds gaussian noise to simulate the real VL53L4CD sensor.
        Returns (measurements, hit_points) where measurements[i] is None for disabled sensors.
        """
        measurements = []
        hit_points = []
        robot_angle = self.angle_deg

        for i, sensor_angle in enumerate(SENSOR_ANGLES):
            if not self.active_sensors[i]:
                measurements.append(None)
                continue

            total_angle = (robot_angle + sensor_angle) % 360
            rad = math.radians(total_angle)
            dx = math.cos(rad)
            dy = math.sin(rad)

            curr_x, curr_y = self.true_x, self.true_y
            dist_px = 0
            hit = False
            step_size = 2.0

            while dist_px < MAX_RANGE_PX:
                next_x = curr_x + dx * step_size
                next_y = curr_y + dy * step_size

                curr_c_cell = int(curr_x // GRID_SIZE)
                curr_r_cell = int(curr_y // GRID_SIZE)
                next_c_cell = int(next_x // GRID_SIZE)
                next_r_cell = int(next_y // GRID_SIZE)

                if not (0 <= next_c_cell < MAP_COLS and 0 <= next_r_cell < MAP_ROWS):
                    hit = True
                    break

                if next_c_cell != curr_c_cell:
                    wall = WALL_RIGHT if next_c_cell > curr_c_cell else WALL_LEFT
                    if wall_map[curr_r_cell, curr_c_cell, wall] == 0:
                        hit = True
                        break
                if next_r_cell != curr_r_cell:
                    wall = WALL_DOWN if next_r_cell > curr_r_cell else WALL_UP
                    if wall_map[curr_r_cell, curr_c_cell, wall] == 0:
                        hit = True
                        break

                curr_x, curr_y = next_x, next_y
                dist_px += step_size

            dist_mm = dist_px / PIXELS_PER_MM
            dist_mm = max(MIN_RANGE_MM, min(dist_mm, MAX_RANGE_MM))

            if add_noise and hit:
                dist_mm = random.gauss(dist_mm, SENSOR_MEASUREMENT_SIGMA_MM)
                dist_mm = max(MIN_RANGE_MM, min(dist_mm, MAX_RANGE_MM))

            measurements.append(dist_mm)
            hit_points.append((curr_x, curr_y))

        return measurements, hit_points
