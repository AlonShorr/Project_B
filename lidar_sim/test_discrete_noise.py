"""Headless smoke checks for the discrete stochastic localization model.

Run from this directory:
    python3 test_discrete_noise.py
"""

import numpy as np

import config
from config import (
    MAP_COLS,
    MAP_ROWS,
    CELL_SIZE_MM,
    ROTATION_STEP,
    FORWARD_SIGMA_MM,
    FORWARD_TARGET_MM,
    FORWARD_TRANSITION_BIN_HALF_WIDTH_MM,
    TURN_SIGMA_DEG,
    TURN_TARGET_DEG,
    TURN_TRANSITION_BIN_HALF_WIDTH_DEG,
)
from discrete_noise import (
    forward_transition_probs,
    gaussian_three_bin_probs,
    turn_transition_probs,
)
from localization import (
    collapse_position_belief,
    initialize_belief,
    predict_action,
    predict_forward,
    predict_turn,
    update_probability,
)
from map_builder import generate_wall_map
from motion_planning import plan_path


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"[OK] {name}")


def main():
    wall_map = generate_wall_map()

    forward_probs = gaussian_three_bin_probs(
        FORWARD_TARGET_MM, FORWARD_TRANSITION_BIN_HALF_WIDTH_MM, FORWARD_SIGMA_MM
    )
    turn_probs = gaussian_three_bin_probs(
        TURN_TARGET_DEG, TURN_TRANSITION_BIN_HALF_WIDTH_DEG, TURN_SIGMA_DEG
    )
    check("forward probabilities sum to 1", abs(sum(forward_probs) - 1.0) < 1e-12)
    check("turn probabilities sum to 1", abs(sum(turn_probs) - 1.0) < 1e-12)
    check("default forward bin gives high p_correct", forward_probs[1] > 0.999)
    check("default turn bin gives high p_correct", turn_probs[1] > 0.999)

    before_forward = forward_transition_probs(noisy=True)
    before_turn = turn_transition_probs(noisy=True)
    old_forward_control = config.FORWARD_CONTROL_TOLERANCE_MM
    old_turn_control = config.TURN_CONTROL_TOLERANCE_DEG
    config.FORWARD_CONTROL_TOLERANCE_MM = 1.0
    config.TURN_CONTROL_TOLERANCE_DEG = 1.0
    try:
        check("control tolerance does not change forward transition", forward_transition_probs(noisy=True) == before_forward)
        check("control tolerance does not change turn transition", turn_transition_probs(noisy=True) == before_turn)
    finally:
        config.FORWARD_CONTROL_TOLERANCE_MM = old_forward_control
        config.TURN_CONTROL_TOLERANCE_DEG = old_turn_control

    old_forward_bin = config.FORWARD_TRANSITION_BIN_HALF_WIDTH_MM
    old_turn_bin = config.TURN_TRANSITION_BIN_HALF_WIDTH_DEG
    config.FORWARD_TRANSITION_BIN_HALF_WIDTH_MM = CELL_SIZE_MM / 18.0
    config.TURN_TRANSITION_BIN_HALF_WIDTH_DEG = ROTATION_STEP / 18.0
    try:
        check("forward transition bin changes probabilities", forward_transition_probs(noisy=True)[1] < before_forward[1])
        check("turn transition bin changes probabilities", turn_transition_probs(noisy=True)[1] < before_turn[1])
    finally:
        config.FORWARD_TRANSITION_BIN_HALF_WIDTH_MM = old_forward_bin
        config.TURN_TRANSITION_BIN_HALF_WIDTH_DEG = old_turn_bin

    belief = initialize_belief(MAP_ROWS, MAP_COLS)
    check("initialize_belief returns (rows, cols, 8)", belief.shape == (MAP_ROWS, MAP_COLS, 8))
    check("initialize_belief sums to 1", abs(float(belief.sum()) - 1.0) < 1e-12)

    expected = np.full((MAP_ROWS, MAP_COLS, 8, 5), 180.0)
    measured = [180.0, 180.0, 180.0, 180.0, 180.0]
    corrected = update_probability(belief, measured, expected)
    check("update_probability keeps probability normalized", abs(float(corrected.sum()) - 1.0) < 1e-12)

    turned = predict_turn(corrected, 1, noisy=True)
    check("predict_turn keeps probability normalized", abs(float(turned.sum()) - 1.0) < 1e-12)

    moved = predict_forward(turned, wall_map, noisy=True)
    check("predict_forward keeps probability normalized", abs(float(moved.sum()) - 1.0) < 1e-12)
    action_predicted = predict_action(moved, "FORWARD", wall_map, noisy=True)
    check("predict_action keeps probability normalized", abs(float(action_predicted.sum()) - 1.0) < 1e-12)
    position = collapse_position_belief(action_predicted)
    check("collapse_position_belief returns (rows, cols)", position.shape == (MAP_ROWS, MAP_COLS))
    check("collapse_position_belief sums to 1", abs(float(position.sum()) - 1.0) < 1e-12)

    path = plan_path((3, 3, 0), (0, 9), wall_map)
    check("A* still returns a path", len(path) > 0)

    print("Discrete noise smoke checks passed.")


if __name__ == "__main__":
    main()
