"""Active localization: greedy one-step (myopic) information-gain action selection.

This module decides the *next exploratory action* for the robot while it is lost
or its belief is multimodal (several cells sharing the same occupancy
probability). Passive re-measurement cannot break such ties: every competing
hypothesis predicts the same reading, so the belief stays multimodal forever.
Active localization deliberately picks the action whose *expected* observation
differs most across the competing hypotheses, collapsing the belief fastest.

Theory
------
The belief ``Bel`` is the probability matrix maintained by ``localization.py``.
Its uncertainty is measured by Shannon entropy (in nats):

    H(Bel) = - sum_x Bel(x) * ln Bel(x)

For ``k`` equiprobable cells, H = ln k (maximal); a fully localized belief has
H = 0. Each action ``a`` should reduce H. Reasoning one step ahead:

  1. Prediction step  -> Bel^-(x') = sum_x p(x'|x,a) Bel(x)
     (rotations do not translate, so Bel^- = Bel; for moves we reuse the
     simulator's ``predict_motion`` so the lookahead matches the real model,
     blur included).
  2. Correction step  -> for a hypothetical reading z, Bel^+_z = update(Bel^-, z).

The expected posterior entropy of action ``a`` averages over the readings the
robot might actually obtain:

    E_z[H | a] = sum_z p(z | a, Bel^-) * H(Bel^+_z)

and we choose  a* = argmax_a [ H(Bel) - E_z[H | a] - cost(a) ].

The integral over all readings is approximated, as is standard for a discrete
grid with a known map, by treating the precomputed noiseless reading at each
plausible pose as a representative observation, weighted by belief:

    E_z[H | a] ~= sum_{x'} Bel^-(x') * H( update(Bel^-, z_hat[x'], theta_a) )

where ``z_hat[x'] = expected_data[x'][theta_a]`` is exactly the table built by
``precompute_all_orientations``. Intuitively: "for each cell I might really be
in, simulate the reading I'd get after action a, see how much it would sharpen
my belief, then average over cells weighted by how likely each one is."

References
----------
- Burgard, W., Fox, D., Thrun, S. "Active Mobile Robot Localization."
  Proc. IJCAI 1997, pp. 1346-1352. (Action = argmax of expected entropy
  reduction minus travel cost; also chooses sensor pointing direction.)
- Fox, D., Burgard, W., Thrun, S. "Active Markov Localization for Mobile
  Robots." Robotics and Autonomous Systems, 25(3-4), 1998. (Journal version.)
- Thrun, S., Burgard, W., Fox. "Probabilistic Robotics." MIT Press, 2005.
  (Markov localization; entropy as a belief-uncertainty measure.)
- Cover, T., Thomas, J. "Elements of Information Theory." Wiley, 1991.
  (Shannon entropy.)

Design notes
------------
- Pure functions, no global state; the caller (the pygame loop) applies the
  chosen action to the ``Robot``. This keeps the policy independently testable.
- Sensor pointing in Burgard et al. becomes "which heading to rotate to" here,
  because the five ToF sensors are rigidly mounted. Rotations are first-class,
  cheap actions and often break symmetry without any travel.
- Costs are expressed in the same units as entropy (nats) so they trade off
  directly. Defaults are small relative to a typical ln(k) gain, so during the
  lost/multimodal phase information dominates; raise them to make the robot
  prefer staying put / turning over driving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from config import (
    SENSOR_ANGLES,
    WALL_UP, WALL_DOWN, WALL_LEFT, WALL_RIGHT,
    DIR_TO_ANGLE,
)
from localization import predict_forward, predict_turn, update_probability

# Cardinal moves the simulator/RRT planner uses, paired with the wall flag that
# blocks them and the resulting heading index. (dr, dc) -> (wall_idx, heading).
_CARDINAL_MOVES = {
    (-1, 0): (WALL_UP, DIR_TO_ANGLE[(-1, 0)]),
    (1, 0): (WALL_DOWN, DIR_TO_ANGLE[(1, 0)]),
    (0, -1): (WALL_LEFT, DIR_TO_ANGLE[(0, -1)]),
    (0, 1): (WALL_RIGHT, DIR_TO_ANGLE[(0, 1)]),
}

_N_HEADINGS = 8


@dataclass(frozen=True)
class Action:
    """A candidate one-step action and how the caller should execute it.

    Attributes:
        kind: ``"rotate"`` (turn in place) or ``"move"`` (turn-then-step).
        dr: Row delta for a move; ``0`` for a rotation.
        dc: Column delta for a move; ``0`` for a rotation.
        heading: Resulting heading index (0=E, 2=S, 4=W, 6=N, ...) at which the
            robot will observe after the action.
        n_turns: Number of 45-degree rotation steps needed to reach ``heading``
            from the current heading (shortest direction).
        turn_direction: +1 clockwise or -1 counter-clockwise for the shortest
            turn sequence to ``heading``.
        label: Human-readable description for logging.
    """

    kind: str
    dr: int
    dc: int
    heading: int
    n_turns: int
    turn_direction: int
    label: str


def _turn_steps(a: int, b: int) -> int:
    """Minimum number of 45-degree steps to rotate from heading ``a`` to ``b``."""
    diff = (a - b) % _N_HEADINGS
    return min(diff, _N_HEADINGS - diff)


def belief_entropy(belief: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a belief distribution, in nats.

    H = -sum_x p(x) ln p(x), summed over cells with p > eps. A peaked (well
    localized) belief returns ~0; a uniform belief over k cells returns ln k.

    Args:
        belief: Probability matrix (need not be exactly normalized; it is
            renormalized defensively before the entropy is computed).
        eps: Cells with probability at or below this are treated as zero
            (0 * ln 0 = 0) to avoid NaNs.

    Returns:
        Entropy in nats (non-negative).
    """
    p = np.asarray(belief, dtype=float)
    total = p.sum()
    if total <= 0.0:
        return 0.0
    p = p / total
    mask = p > eps
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _top_hypotheses(
    belief: np.ndarray,
    prob_floor: float,
    max_hypotheses: int,
) -> list[tuple[int, int, int, float]]:
    """Return the most probable states as ``(row, col, theta, weight)``.

    Only these "competing hypotheses" matter for breaking a multimodal belief,
    so restricting the expected-entropy average to the top-K keeps the policy
    fast without changing its decisions in the cases it is meant for.
    """
    flat = np.asarray(belief, dtype=float).ravel()
    order = np.argsort(flat)[::-1]
    out: list[tuple[int, int, int, float]] = []
    for idx in order:
        w = float(flat[idx])
        if w < prob_floor:
            break
        if belief.ndim == 3:
            r, c, theta = np.unravel_index(int(idx), belief.shape)
        else:
            r, c = divmod(int(idx), belief.shape[1])
            theta = 0
        out.append((int(r), int(c), int(theta), w))
        if len(out) >= max_hypotheses:
            break
    return out


def expected_entropy_after_action(
    predicted_belief: np.ndarray,
    observe_heading: int,
    expected_data: np.ndarray,
    prob_floor: float = 1e-4,
    max_hypotheses: int = 16,
) -> float:
    """Expected belief entropy after observing at ``observe_heading``.

    Implements the weighted-hypothesis approximation of E_z[H | a]: for each
    plausible pose ``x'`` (taken from the predicted belief), use that pose's
    precomputed noiseless reading as the hypothetical observation, run the
    correction step, and average the resulting posterior entropy weighted by
    Bel^-(x').

    Args:
        predicted_belief: Bel^- after the motion model has been applied
            (equal to the current belief for in-place rotations).
        observe_heading: Heading index the robot will face when it measures.
        expected_data: Precomputed table from ``precompute_all_orientations``,
            shape ``(rows, cols, 8, n_sensors)``.
        prob_floor: Hypotheses below this probability are ignored.
        max_hypotheses: Cap on the number of hypotheses averaged over.

    Returns:
        Expected posterior entropy in nats. Falls back to the entropy of
        ``predicted_belief`` if no hypothesis clears ``prob_floor``.
    """
    hypotheses = _top_hypotheses(predicted_belief, prob_floor, max_hypotheses)
    if not hypotheses:
        return belief_entropy(predicted_belief)

    weighted_entropy = 0.0
    weight_sum = 0.0
    for r, c, theta, w in hypotheses:
        hypothetical_reading = expected_data[r, c, theta if predicted_belief.ndim == 3 else observe_heading]
        if predicted_belief.ndim == 3:
            posterior = update_probability(predicted_belief, hypothetical_reading, expected_data)
        else:
            posterior = update_probability(
                predicted_belief, hypothetical_reading, expected_data, observe_heading
            )
        weighted_entropy += w * belief_entropy(posterior)
        weight_sum += w

    if weight_sum <= 0.0:
        return belief_entropy(predicted_belief)
    return weighted_entropy / weight_sum


def information_gain(
    belief: np.ndarray,
    action: Action,
    expected_data: np.ndarray,
    wall_map: np.ndarray,
    prob_floor: float = 1e-4,
    max_hypotheses: int = 16,
) -> tuple[float, float, float]:
    """Expected entropy reduction from taking ``action`` then observing.

    Returns:
        ``(gain, current_entropy, expected_entropy)`` where
        ``gain = current_entropy - expected_entropy`` (in nats). The motion
        model's blur can *raise* entropy on a move, so for a pure move the gain
        nets the blur-induced increase against the observation's decrease; a
        rotation has no blur and so isolates the observation's benefit.
    """
    current_entropy = belief_entropy(belief)

    predicted = belief
    for _ in range(action.n_turns):
        predicted = predict_turn(predicted, action.turn_direction, noisy=True)

    if action.kind == "move":
        predicted = predict_forward(predicted, wall_map, noisy=True)

    expected = expected_entropy_after_action(
        predicted, action.heading, expected_data, prob_floor, max_hypotheses
    )
    return current_entropy - expected, current_entropy, expected


def _enumerate_actions(current_heading: int) -> list[Action]:
    """Build the candidate action set: rotate to each other heading + 4 moves.

    Rotating to the current heading is omitted (re-observing the same pose adds
    no new information). Moves are always considered in all four cardinal
    directions; ``predict_motion`` handles the per-hypothesis wall blocking, so
    a move is meaningful even when some hypotheses are walled in.
    """
    actions: list[Action] = []

    for heading in range(_N_HEADINGS):
        if heading == current_heading:
            continue
        delta = (heading - current_heading) % _N_HEADINGS
        actions.append(
            Action(
                kind="rotate",
                dr=0,
                dc=0,
                heading=heading,
                n_turns=_turn_steps(current_heading, heading),
                turn_direction=1 if delta <= 4 else -1,
                label=f"rotate to heading {heading} ({heading * 45} deg)",
            )
        )

    for (dr, dc), (_wall_idx, heading) in _CARDINAL_MOVES.items():
        delta = (heading - current_heading) % _N_HEADINGS
        actions.append(
            Action(
                kind="move",
                dr=dr,
                dc=dc,
                heading=heading,
                n_turns=_turn_steps(current_heading, heading),
                turn_direction=1 if delta <= 4 else -1,
                label=f"move (dr={dr}, dc={dc}) facing heading {heading}",
            )
        )

    return actions


def select_best_action(
    belief: np.ndarray,
    expected_data: np.ndarray,
    wall_map: np.ndarray,
    current_heading: int,
    move_cost: float = 0.10,
    turn_cost: float = 0.02,
    prob_floor: float = 1e-4,
    max_hypotheses: int = 16,
) -> dict[str, Any]:
    """Greedy one-step active-localization policy.

    Scores every candidate action by expected information gain minus its cost
    and returns the best one. A one-step lookahead is enough to
    break the symmetric beliefs.

    Args:
        belief: Current probability matrix, shape ``(rows, cols)``.
        expected_data: Table from ``precompute_all_orientations``.
        wall_map: Map array, shape ``(rows, cols, 4)``.
        current_heading: Robot's current heading index (``robot.angle_index``).
        move_cost: Cost (nats) charged for taking a forward step.
        turn_cost: Cost (nats) charged per 45-degree rotation step.
        prob_floor: Hypotheses below this probability are ignored.
        max_hypotheses: Cap on hypotheses in the expected-entropy average.

    Returns:
        A dict with:
            ``"action"``: the chosen :class:`Action` (or ``None`` if the belief
                is already effectively localized / degenerate).
            ``"gain"``: expected entropy reduction of the chosen action (nats).
            ``"utility"``: gain minus action cost (the maximized quantity).
            ``"current_entropy"``: entropy of the current belief (nats).
            ``"expected_entropy"``: expected posterior entropy of the choice.
            ``"ranked"``: all candidates as dicts, best first (for debugging).
            ``"reason"``: short human-readable explanation.
    """
    current_entropy = belief_entropy(belief)

    if current_entropy <= 1e-6:
        return {
            "action": None,
            "gain": 0.0,
            "utility": 0.0,
            "current_entropy": current_entropy,
            "expected_entropy": current_entropy,
            "ranked": [],
            "reason": "belief already localized (entropy ~ 0)",
        }

    ranked: list[dict[str, Any]] = []
    for action in _enumerate_actions(current_heading):
        gain, cur_h, exp_h = information_gain(
            belief, action, expected_data, wall_map, prob_floor, max_hypotheses
        )
        cost = move_cost * (1 if action.kind == "move" else 0) + turn_cost * action.n_turns
        utility = gain - cost
        ranked.append(
            {
                "action": action,
                "gain": gain,
                "cost": cost,
                "utility": utility,
                "expected_entropy": exp_h,
            }
        )

    ranked.sort(key=lambda item: item["utility"], reverse=True)
    best = ranked[0]
    return {
        "action": best["action"],
        "gain": best["gain"],
        "utility": best["utility"],
        "current_entropy": current_entropy,
        "expected_entropy": best["expected_entropy"],
        "ranked": ranked,
        "reason": (
            f"chose '{best['action'].label}': expected entropy "
            f"{current_entropy:.3f} -> {best['expected_entropy']:.3f} nats "
            f"(gain {best['gain']:.3f}, cost {best['cost']:.3f})"
        ),
    }


def apply_action_to_robot(
    robot: Any,
    action: Optional[Action],
    wall_map: np.ndarray,
    rng: Any = None,
    noisy: bool = False,
) -> tuple[int, int, Optional[str]]:
    """Convenience executor: turn the robot toward ``action`` and step if a move.

    Mirrors the one-action-per-frame style of ``do_localization_step`` so it can
    drop into the ``AUTO_LOCALIZING`` branch of the pygame loop. Rotates one
    45-degree step toward the target heading; only once aligned does a ``move``
    action actually step forward (so the caller sees the turn animate first).

    Args:
        robot: A ``Robot`` instance (mutated in place).
        action: The action chosen by :func:`select_best_action`, or ``None``.
        wall_map: Map array used by ``robot.move_rel`` for wall checking.

    Returns:
        ``(dr, dc, commanded_action)`` for the frame.
    """
    if action is None:
        return 0, 0, None

    if robot.angle_index != action.heading:
        delta = (action.heading - robot.angle_index) % _N_HEADINGS
        direction = 1 if delta <= 4 else -1
        command = "TURN_RIGHT" if direction > 0 else "TURN_LEFT"
        if noisy and rng is not None:
            robot.apply_action(command, wall_map, rng, noisy=True)
        else:
            robot.rotate(direction)
        return 0, 0, command

    if action.kind == "move":
        old_r, old_c = robot.r, robot.c
        if noisy and rng is not None:
            robot.apply_action("FORWARD", wall_map, rng, noisy=True)
        else:
            robot.move_rel(action.dr, action.dc, wall_map)
        return robot.r - old_r, robot.c - old_c, "FORWARD"
    return 0, 0, None


# ---------------------------------------------------------------------------
# Self-test / demo. Runs headless against the real filter and map modules.
#   python3 active_localization.py
# ---------------------------------------------------------------------------
def _demo() -> int:
    import numpy as _np
    from map_builder import generate_wall_map
    from localization import initialize_belief, precompute_all_orientations

    wall_map = generate_wall_map()
    rows, cols, _ = wall_map.shape
    expected_data = precompute_all_orientations(wall_map)

    failures = 0

    def check(name: str, ok: bool) -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print("\n--- entropy sanity ---")
    k = 4
    b = _np.zeros((rows, cols, _N_HEADINGS))
    cells = [(3, 3), (3, 5), (5, 3), (5, 5)]
    for (r, c) in cells:
        b[r, c, 0] = 1.0 / k
    check("uniform over k cells gives H = ln k",
          abs(belief_entropy(b) - _np.log(k)) < 1e-9)
    spike = _np.zeros((rows, cols, _N_HEADINGS))
    spike[7, 7, 0] = 1.0
    check("a localized spike gives H ~ 0", belief_entropy(spike) < 1e-9)

    print("\n--- uniform belief: an action with positive gain exists ---")
    uniform = initialize_belief(rows, cols)
    res = select_best_action(uniform, expected_data, wall_map, current_heading=2)
    print("  " + res["reason"])
    check("best action defined", res["action"] is not None)
    check("expected entropy < current entropy",
          res["expected_entropy"] < res["current_entropy"])
    check("information gain positive", res["gain"] > 0)

    print("\n--- multimodal belief: policy reduces expected entropy ---")
    # Place equal mass on two interior cells and verify the chosen action is
    # expected to collapse the belief.
    multi = _np.zeros((rows, cols, _N_HEADINGS))
    for (r, c) in [(6, 6), (9, 12)]:
        multi[r, c, 0] = 0.5
    res2 = select_best_action(multi, expected_data, wall_map, current_heading=0)
    print("  " + res2["reason"])
    check("two-mode start entropy ~ ln 2",
          abs(res2["current_entropy"] - _np.log(2)) < 0.05)
    check("chosen action lowers expected entropy",
          res2["expected_entropy"] < res2["current_entropy"])

    print("\n--- rotations are scored (sensor-pointing decision) ---")
    has_rotate = any(item["action"].kind == "rotate" for item in res["ranked"])
    check("rotate actions present in candidate set", has_rotate)

    print("\n--- top-3 candidates for the uniform case ---")
    for item in res["ranked"][:3]:
        a = item["action"]
        print(f"    {a.label:<42} utility={item['utility']:+.3f} "
              f"gain={item['gain']:+.3f} cost={item['cost']:.3f}")

    print(f"\nDemo {'PASSED' if failures == 0 else f'FAILED ({failures})'}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_demo())
