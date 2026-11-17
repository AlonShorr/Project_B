import numpy as np
import random

from config import (
    MAP_COLS, SENSOR_ANGLES, MIN_RANGE_MM,
    WALL_UP, WALL_DOWN, WALL_LEFT, WALL_RIGHT,
    LOCALIZATION_MOVES, SENSOR_LIKELIHOOD_SIGMA_MM,
    SENSOR_LIKELIHOOD_FLOOR, SENSOR_LIKELIHOOD_POWER,
)
from discrete_noise import forward_transition_probs, turn_transition_probs
from sim_robot import Robot

N_ORIENTATIONS = 8
ACTION_FORWARD = "FORWARD"
ACTION_TURN_LEFT = "TURN_LEFT"
ACTION_TURN_RIGHT = "TURN_RIGHT"

THETA_TO_STEP = {
    0: (0, 1, WALL_RIGHT),
    2: (1, 0, WALL_DOWN),
    4: (0, -1, WALL_LEFT),
    6: (-1, 0, WALL_UP),
}


def initialize_belief(rows, cols):
    """Return a uniform belief over (row, col, orientation)."""
    belief = np.ones((rows, cols, N_ORIENTATIONS), dtype=float)
    belief /= belief.size
    return belief


def precompute_all_orientations(wall_map):
    """
    Pre-compute noiseless sensor readings for every (row, col, angle) combination.
    Returns ndarray of shape (Rows, Cols, 8 angles, N sensors).
    """
    print("Pre-computing for 8 orientations... (Wait for it)")
    rows, cols, _ = wall_map.shape
    data = np.zeros((rows, cols, N_ORIENTATIONS, len(SENSOR_ANGLES)))

    v_bot = Robot(0, 0)
    for r in range(rows):
        for c in range(cols):
            v_bot.move_to(r, c)
            for ang_idx in range(N_ORIENTATIONS):
                v_bot.angle_index = ang_idx
                v_bot.active_sensors = [True] * len(SENSOR_ANGLES)
                dists, _ = v_bot.measure(wall_map)
                data[r, c, ang_idx] = dists

    print("Pre-computation Complete.")
    return data


def _normalize_or_uniform(prob):
    total = float(np.sum(prob))
    if total > 0.0:
        return prob / total
    out = np.ones_like(prob, dtype=float)
    out /= out.size
    return out


def update_probability(prob_matrix, measured_dists, expected_data):
    """
    Bayes correction step over every discrete (row, col, theta) state.
    """
    rows, cols, orientations = prob_matrix.shape
    new_prob = np.zeros_like(prob_matrix)
    sigma = SENSOR_LIKELIHOOD_SIGMA_MM

    for r in range(rows):
        for c in range(cols):
            for theta in range(orientations):
                prior = prob_matrix[r, c, theta]
                if prior < SENSOR_LIKELIHOOD_FLOOR:
                    continue

                expected_dists = expected_data[r, c, theta]
                likelihood = 1.0
                for i in range(len(SENSOR_ANGLES)):
                    z = measured_dists[i]
                    if z is None:
                        continue
                    mu = max(expected_dists[i], MIN_RANGE_MM)
                    likelihood *= np.exp(-((z - mu) ** 2) / (2 * sigma ** 2))

                likelihood = max(float(likelihood), SENSOR_LIKELIHOOD_FLOOR)
                if SENSOR_LIKELIHOOD_POWER != 1.0:
                    likelihood = likelihood ** SENSOR_LIKELIHOOD_POWER
                new_prob[r, c, theta] = prior * likelihood

    return _normalize_or_uniform(new_prob)


def _advance_one_cell(r, c, theta, wall_map):
    move = THETA_TO_STEP.get(theta % N_ORIENTATIONS)
    if move is None:
        return r, c

    rows, cols = wall_map.shape[:2]
    dr, dc, wall_idx = move
    nr, nc = r + dr, c + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return r, c
    if wall_map[r, c, wall_idx] == 0:
        return r, c
    return nr, nc


def predict_forward(prob_matrix, wall_map, noisy=None):
    """Prediction for a commanded one-cell forward move."""
    rows, cols, orientations = prob_matrix.shape
    predicted = np.zeros_like(prob_matrix)
    probs = forward_transition_probs(noisy)

    for r in range(rows):
        for c in range(cols):
            for theta in range(orientations):
                mass = prob_matrix[r, c, theta]
                if mass <= SENSOR_LIKELIHOOD_FLOOR:
                    continue
                if theta not in THETA_TO_STEP:
                    predicted[r, c, theta] += mass
                    continue

                # undershoot: stay
                predicted[r, c, theta] += mass * probs[0]

                # correct: one reachable cell, or the last reachable state
                r1, c1 = _advance_one_cell(r, c, theta, wall_map)
                predicted[r1, c1, theta] += mass * probs[1]

                # overshoot: cross the first and then second edge if possible
                r2, c2 = _advance_one_cell(r1, c1, theta, wall_map)
                predicted[r2, c2, theta] += mass * probs[2]

    return _normalize_or_uniform(predicted)


def predict_turn(prob_matrix, direction, noisy=None):
    """Prediction for a commanded +/- one-index turn."""
    direction = 1 if direction > 0 else -1
    rows, cols, orientations = prob_matrix.shape
    predicted = np.zeros_like(prob_matrix)
    probs = turn_transition_probs(noisy)

    for r in range(rows):
        for c in range(cols):
            for theta in range(orientations):
                mass = prob_matrix[r, c, theta]
                if mass <= SENSOR_LIKELIHOOD_FLOOR:
                    continue
                predicted[r, c, theta] += mass * probs[0]
                predicted[r, c, (theta + direction) % orientations] += mass * probs[1]
                predicted[r, c, (theta + 2 * direction) % orientations] += mass * probs[2]

    return _normalize_or_uniform(predicted)


def predict_action(prob_matrix, action, wall_map, noisy=None):
    """Dispatch prediction for FORWARD, TURN_LEFT, or TURN_RIGHT."""
    if action == ACTION_FORWARD:
        return predict_forward(prob_matrix, wall_map, noisy)
    if action == ACTION_TURN_LEFT:
        return predict_turn(prob_matrix, -1, noisy)
    if action == ACTION_TURN_RIGHT:
        return predict_turn(prob_matrix, 1, noisy)
    raise ValueError(f"unknown action: {action}")


def predict_motion(prob_matrix, dr, dc, wall_map, noisy=None):
    """Compatibility wrapper for old cardinal move callers."""
    theta_lookup = {(-1, 0): 6, (1, 0): 2, (0, -1): 4, (0, 1): 0}
    theta = theta_lookup.get((dr, dc))
    if theta is None:
        return prob_matrix
    directed = np.zeros_like(prob_matrix)
    directed[:, :, theta] = np.sum(prob_matrix, axis=2)
    return predict_forward(directed, wall_map, noisy)


def get_best_estimate(prob_matrix):
    """
    Return (r, c, theta, peak_probability) for the most likely state
    where state is (position, orientation).
    """
    idx = np.argmax(prob_matrix)
    r, c, theta = np.unravel_index(idx, prob_matrix.shape)
    return int(r), int(c), int(theta), float(prob_matrix[r, c, theta])


def collapse_position_belief(prob_matrix):
    """
    Return position (ONLY) probability matrix by summing out orientations
    for each cell in the original probability matrix.
    """
    return np.sum(prob_matrix, axis=2)


def dominant_position_modes(prob_matrix, min_separation=2):
    """
    In case 2 cells have the same occupency probabilty
    and they're not adjacent, this function will return both 
    of the positions as peak and runner-up - rival cell of the
    peak probability cell. 
    """
    # get position probability matrix
    pos_prob_matrix = collapse_position_belief(prob_matrix) 

    # get first high-occ-prob cell
    pr, pc = np.unravel_index(int(np.argmax(pos_prob_matrix)), pos_prob_matrix.shape)
    peak = float(pos_prob_matrix[pr, pc]) 

    # make a mask to rule out adjacent cells from the peak cell
    rows, cols = pos_prob_matrix.shape
    rr = np.arange(rows)[:, None]
    cc = np.arange(cols)[None, :]
    far = (np.abs(rr - pr) >= min_separation) | (np.abs(cc - pc) >= min_separation)

    # get second high-occ-prob cell which isn't adjacent to the first one (if exists)
    runner_up = float(pos_prob_matrix[far].max()) if far.any() else 0.0
    return peak, runner_up


def do_localization_step(robot, wall_map, spin_count, move_count):
    """
    Advance the localization routine by one action frame:
    first complete a full 360° spin (8 steps), then try random moves.
    Returns (spin_count, move_count, moved_dr, moved_dc).
    """
    RANDOM_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    moved_dr, moved_dc = 0, 0

    if spin_count < 8:
        robot.rotate(1)
        spin_count += 1
    elif move_count < LOCALIZATION_MOVES:
        dr, dc = random.choice(RANDOM_DIRS)
        if robot.move_rel(dr, dc, wall_map):
            moved_dr, moved_dc = dr, dc
        move_count += 1
    else:
        spin_count = 0
        move_count = 0

    return spin_count, move_count, moved_dr, moved_dc
