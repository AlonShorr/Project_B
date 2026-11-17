"""Console verification demo for the standalone RRT planner.

The official planner output remains ``result["path"]`` as ``[(r, c), ...]``.
Debug actions are printed and replayed only so humans can verify the route.
"""

from __future__ import annotations

from pathlib import Path
import sys

from rrt_planner import (
    FORWARD,
    TURN_AROUND,
    TURN_LEFT,
    TURN_RIGHT,
    apply_internal_action,
    can_move_forward,
    path_cells_to_world,
    plan_rrt,
    validate_path,
)


class DemoWallMap(list):
    @property
    def shape(self):
        rows = len(self)
        cols = len(self[0]) if rows else 0
        return rows, cols, 4


def wall_at(wall_map, r, c, wall_idx):
    return int(wall_map[r][c][wall_idx])


def validate_wall_map_compatibility(wall_map):
    if not hasattr(wall_map, "shape") or len(wall_map.shape) != 3 or wall_map.shape[2] < 4:
        return False, "wall_map must have shape (rows, cols, 4)"

    rows, cols, _ = wall_map.shape
    if rows <= 0 or cols <= 0:
        return False, "wall_map must contain at least one row and one column"

    for r in range(rows):
        for c in range(cols):
            for wall_idx in range(4):
                if wall_at(wall_map, r, c, wall_idx) not in (0, 1):
                    return False, f"wall_map[{r}][{c}][{wall_idx}] is not 0 or 1"

            if r == 0 and wall_at(wall_map, r, c, 0) != 0:
                return False, f"top border missing at cell {(r, c)}"
            if r == rows - 1 and wall_at(wall_map, r, c, 1) != 0:
                return False, f"bottom border missing at cell {(r, c)}"
            if c == 0 and wall_at(wall_map, r, c, 2) != 0:
                return False, f"left border missing at cell {(r, c)}"
            if c == cols - 1 and wall_at(wall_map, r, c, 3) != 0:
                return False, f"right border missing at cell {(r, c)}"

            if r + 1 < rows and wall_at(wall_map, r, c, 1) != wall_at(wall_map, r + 1, c, 0):
                return False, f"down/up wall mismatch between {(r, c)} and {(r + 1, c)}"
            if c + 1 < cols and wall_at(wall_map, r, c, 3) != wall_at(wall_map, r, c + 1, 2):
                return False, f"right/left wall mismatch between {(r, c)} and {(r, c + 1)}"

    return True, "compatible with simulator wall map representation"


def build_fallback_wall_map():
    """Build a dependency-free map with internal walls and one-cell passages."""
    rows, cols = 8, 8
    maze = DemoWallMap([[[1, 1, 1, 1] for _ in range(cols)] for _ in range(rows)])

    def add_v_wall(r, c):
        maze[r][c][3] = 0
        if c + 1 < cols:
            maze[r][c + 1][2] = 0

    def add_h_wall(r, c):
        maze[r][c][1] = 0
        if r + 1 < rows:
            maze[r + 1][c][0] = 0

    for c in range(cols):
        maze[0][c][0] = 0
        maze[rows - 1][c][1] = 0
    for r in range(rows):
        maze[r][0][2] = 0
        maze[r][cols - 1][3] = 0

    # Horizontal barrier with two gaps.
    for c in range(cols):
        if c not in (1, 6):
            add_h_wall(2, c)

    # Vertical barrier with two gaps.
    for r in range(3, rows):
        if r not in (4, 7):
            add_v_wall(r, 3)

    # A small pocket near the upper-right to make wall signatures less uniform.
    add_v_wall(0, 5)
    add_h_wall(1, 5)
    add_v_wall(1, 6)
    return maze


def load_wall_map():
    from map_builder import generate_wall_map

    return generate_wall_map(), "real Pygame simulator map", "map_builder.generate_wall_map"


def require_robot_sim_venv():
    expected_prefix = Path(__file__).resolve().parent / "robot_sim"
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix != expected_prefix:
        print(f"Wrong Python environment: {actual_prefix}")
        print(f"Expected robot_sim venv: {expected_prefix}")
        print("Run:")
        print(f"  cd {Path(__file__).resolve().parent}")
        print("  ./robot_sim/bin/python rrt_demo.py")
        raise SystemExit(2)


def plot_xy_for_cell(cell):
    r, c = cell
    return c, r


def action_label(action):
    return action if action is not None else "START"


def replay_debug_actions(wall_map, start, actions):
    state = (int(start[0]), int(start[1]), int(start[2]) if len(start) >= 3 else 0)
    states = [state]
    transitions = []
    for action in actions:
        next_state = apply_internal_action(wall_map, state, action)
        if next_state is None:
            return states, transitions, False, f"{state} --{action}--> blocked"
        transitions.append((state, action, next_state))
        states.append(next_state)
        state = next_state
    return states, transitions, True, "debug actions replayed"


def compact_cell_path_from_states(states):
    cells = [(r, c) for r, c, _ in states]
    compact = [cells[0]]
    for cell in cells[1:]:
        if cell != compact[-1]:
            compact.append(cell)
    return compact


def verify_result(wall_map, start, goal, result):
    if not result["success"]:
        return False, [f"planner failed: {result['debug'].get('reason')}"], []

    issues = []
    actions = result["debug"].get("debug_actions", [])
    replay_states, transitions, replay_ok, replay_reason = replay_debug_actions(wall_map, start, actions)
    expected_states = [(int(start[0]), int(start[1]), int(start[2]) if len(start) >= 3 else 0)] + result["path_with_theta"]
    full_cell_path = [(int(start[0]), int(start[1]))] + result["path"]
    replay_cell_path = compact_cell_path_from_states(replay_states)
    valid, validation_reason = validate_path(wall_map, start, goal, result["path"])

    if result["path"] and result["path"][0] == (int(start[0]), int(start[1])):
        issues.append("result['path'] includes the start cell")
    if full_cell_path[-1] != (int(goal[0]), int(goal[1])):
        issues.append("final cell does not equal goal cell")
    if not valid:
        issues.append(f"validate_path failed: {validation_reason}")
    if not replay_ok:
        issues.append(replay_reason)
    if replay_states != expected_states:
        issues.append("debug action replay does not match path_with_theta")
    if replay_cell_path != full_cell_path:
        issues.append("debug action replay does not match official path cells")

    for prev, action, nxt in transitions:
        moved = prev[:2] != nxt[:2]
        if action == FORWARD and not moved:
            issues.append(f"FORWARD did not change cells: {prev} -> {nxt}")
        if action != FORWARD and moved:
            issues.append(f"turn action changed cells: {prev} --{action}--> {nxt}")

    return not issues, issues or [validation_reason], transitions


def run_smoke_checks(wall_map, cases):
    print("\nSmoke checks:")
    checks = []
    compatible, compatible_reason = validate_wall_map_compatibility(wall_map)
    checks.append((f"wall map representation compatible ({compatible_reason})", compatible))
    checks.append(("outside map forward fails", not can_move_forward(wall_map, (0, 0, 6))))

    wall_fail = False
    rows, cols, _ = wall_map.shape
    for r in range(rows):
        for c in range(cols):
            for theta in (0, 2, 4, 6):
                if not can_move_forward(wall_map, (r, c, theta)):
                    wall_fail = True
                    break
            if wall_fail:
                break
        if wall_fail:
            break
    checks.append(("moving into at least one wall fails", wall_fail))
    checks.append(("plot coordinates use (c, r)", plot_xy_for_cell((2, 5)) == (5, 2)))

    for idx, (start, goal) in enumerate(cases):
        result = plan_rrt(wall_map, start, goal, max_iter=2500, goal_sample_rate=0.25, random_seed=900 + idx)
        ok, reasons, _ = verify_result(wall_map, start, goal, result)
        checks.append((f"case {idx} reaches goal and respects walls", ok))
        if not ok:
            print(f"  case {idx} detail: {'; '.join(reasons)}")

    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and ok
        print(f"  {name}: {ok}")
    return all_ok


def plot_result_svg(wall_map, result, start, goal, output_path, case_idx, map_source, map_factory, seed):
    rows, cols, _ = wall_map.shape
    cell_px = 42
    margin_left = 54
    margin_top = 42
    width = margin_left + cols * cell_px + 22
    height = margin_top + rows * cell_px + 58

    def x_for_c(c):
        return margin_left + c * cell_px + cell_px / 2

    def y_for_r(r):
        return margin_top + r * cell_px + cell_px / 2

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="22" font-family="Arial" font-size="14">Case {case_idx}: {map_source}; shape={wall_map.shape}; seed={seed}; iterations={result["debug"].get("iterations")}; plotted as (c, r)</text>',
        f'<text x="{margin_left + cols * cell_px / 2 - 36}" y="{height - 14}" font-family="Arial" font-size="12">column c</text>',
        f'<text x="14" y="{margin_top + rows * cell_px / 2}" font-family="Arial" font-size="12" transform="rotate(-90 14 {margin_top + rows * cell_px / 2})">row r</text>',
    ]

    for r in range(rows + 1):
        y = margin_top + r * cell_px
        lines.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + cols * cell_px}" y2="{y}" stroke="#dddddd" stroke-width="1"/>')
    for c in range(cols + 1):
        x = margin_left + c * cell_px
        lines.append(f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{margin_top + rows * cell_px}" stroke="#dddddd" stroke-width="1"/>')

    for r in range(rows):
        lines.append(f'<text x="{margin_left - 24}" y="{y_for_r(r) + 4}" font-family="Arial" font-size="10">{r}</text>')
    for c in range(cols):
        lines.append(f'<text x="{x_for_c(c) - 3}" y="{margin_top - 10}" font-family="Arial" font-size="10">{c}</text>')

    for r in range(rows):
        for c in range(cols):
            x0 = margin_left + c * cell_px
            x1 = x0 + cell_px
            y0 = margin_top + r * cell_px
            y1 = y0 + cell_px
            walls = wall_map[r][c]
            if walls[0] == 0:
                lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black" stroke-width="3"/>')
            if walls[1] == 0:
                lines.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="black" stroke-width="3"/>')
            if walls[2] == 0:
                lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black" stroke-width="3"/>')
            if walls[3] == 0:
                lines.append(f'<line x1="{x1}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="black" stroke-width="3"/>')

    for parent, child in result.get("debug", {}).get("tree_edges", []):
        pr, pc, _ = parent
        cr, cc, _ = child
        lines.append(
            f'<line x1="{x_for_c(pc)}" y1="{y_for_r(pr)}" x2="{x_for_c(cc)}" y2="{y_for_r(cr)}" '
            'stroke="#1f77b4" stroke-width="1" opacity="0.14"/>'
        )

    route = [(start[0], start[1])] + result.get("path", [])
    if len(route) >= 2:
        points = " ".join(f"{x_for_c(c)},{y_for_r(r)}" for r, c in route)
        lines.append(f'<polyline points="{points}" fill="none" stroke="#d62728" stroke-width="4"/>')
        for r, c in route:
            lines.append(f'<circle cx="{x_for_c(c)}" cy="{y_for_r(r)}" r="4" fill="#d62728"/>')

    lines.append(f'<circle cx="{x_for_c(start[1])}" cy="{y_for_r(start[0])}" r="7" fill="#2ca02c"/>')
    lines.append(f'<text x="{x_for_c(start[1]) + 8}" y="{y_for_r(start[0]) - 8}" font-family="Arial" font-size="11">start</text>')
    lines.append(f'<polygon points="{x_for_c(goal[1])},{y_for_r(goal[0]) - 9} {x_for_c(goal[1]) + 8},{y_for_r(goal[0]) + 7} {x_for_c(goal[1]) - 8},{y_for_r(goal[0]) + 7}" fill="#d62728"/>')
    lines.append(f'<text x="{x_for_c(goal[1]) + 8}" y="{y_for_r(goal[0]) - 8}" font-family="Arial" font-size="11">goal</text>')
    lines.append(f'<text x="{margin_left}" y="{height - 34}" font-family="Arial" font-size="10">source: {map_factory}</text>')
    lines.append("</svg>")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"plot saved: {output_path}")
    return output_path


def plot_result(wall_map, result, start, goal, output_path, case_idx, map_source, map_factory, seed):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        svg_path = output_path.with_suffix(".svg")
        print(f"matplotlib unavailable for case {case_idx} ({exc}); writing SVG fallback.")
        return plot_result_svg(wall_map, result, start, goal, svg_path, case_idx, map_source, map_factory, seed)

    rows, cols, _ = wall_map.shape
    fig_w = max(8, cols * 0.52)
    fig_h = max(6, rows * 0.52)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(
        f"Case {case_idx}: {map_source}; shape={wall_map.shape}; "
        f"seed={seed}; iterations={result['debug'].get('iterations')}; plotted as (c, r)"
    )
    ax.set_xlabel("column c")
    ax.set_ylabel("row r")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False)
    ax.grid(which="minor", color="#dddddd", linestyle="-", linewidth=0.8)
    ax.grid(which="major", visible=False)

    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="#e2e2e2",
                    linewidth=0.6,
                    zorder=0,
                )
            )

    for r in range(rows):
        for c in range(cols):
            walls = wall_map[r][c]
            if walls[0] == 0:
                ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], "k-", lw=2.4, solid_capstyle="butt", zorder=3)
            if walls[1] == 0:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], "k-", lw=2.4, solid_capstyle="butt", zorder=3)
            if walls[2] == 0:
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], "k-", lw=2.4, solid_capstyle="butt", zorder=3)
            if walls[3] == 0:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], "k-", lw=2.4, solid_capstyle="butt", zorder=3)

    for parent, child in result.get("debug", {}).get("tree_edges", []):
        pr, pc, _ = parent
        cr, cc, _ = child
        ax.plot([pc, cc], [pr, cr], color="tab:blue", alpha=0.12, lw=0.8, zorder=1)

    route = [(start[0], start[1])] + result.get("path", [])
    if len(route) >= 2:
        xs = [c for r, c in route]
        ys = [r for r, c in route]
        ax.plot(xs, ys, color="tab:red", lw=2.5, marker="o", label="official waypoint path", zorder=4)

    ax.plot(start[1], start[0], "go", markersize=10, label="start", zorder=5)
    ax.plot(goal[1], goal[0], "r*", markersize=14, label="goal", zorder=5)
    ax.text(0.01, 0.01, f"source: {map_factory}", transform=ax.transAxes, fontsize=8, va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"plot saved: {output_path}")
    return output_path


def print_case_report(case_idx, wall_map, map_source, map_factory, start, goal, seed):
    print(f"\nCase {case_idx}:")
    print(f"map source: {map_source} ({map_factory})")
    print(f"start: {start}")
    print(f"goal: {goal}")
    print(f"seed: {seed}")

    result = plan_rrt(
        wall_map=wall_map,
        start=start,
        goal=goal,
        max_iter=2500,
        goal_sample_rate=0.25,
        random_seed=seed,
        return_debug=True,
    )

    debug = result["debug"]
    print(f"success: {result['success']}")
    print(f"iterations: {debug.get('iterations')}")
    print(f"tree nodes: {debug.get('num_nodes')}")
    print(f"path source: {debug.get('path_source')}")
    print(f"raw RRT path cells excluding start: {debug.get('raw_rrt_path')}")

    if not result["success"]:
        print(f"failure reason: {debug.get('reason')}")
        return result, False, []

    full_cell_path = [(start[0], start[1])] + result["path"]
    ok, reasons, transitions = verify_result(wall_map, start, goal, result)

    print(f"path cells excluding start: {result['path']}")
    print(f"full cell path including start: {full_cell_path}")
    print(f"path_with_theta: {result['path_with_theta']}")
    print(f"debug_actions: {debug.get('debug_actions')}")
    print(f"world path: {path_cells_to_world(result['path'])}")
    print(f"path cost: {debug.get('path_cost')}")
    print("step-by-step:")
    for prev, action, nxt in transitions:
        print(f"  state {prev} --{action_label(action)}--> state {nxt}")
    print(f"verification: {ok} ({'; '.join(reasons)})")
    return result, ok, transitions


def main():
    require_robot_sim_venv()
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    wall_map, map_source, map_factory = load_wall_map()
    print(f"Loaded map source: {map_source}")
    print(f"Loaded map factory: {map_factory}")
    print(f"Loaded map shape: {wall_map.shape}")
    compatible, compatible_reason = validate_wall_map_compatibility(wall_map)
    print(f"Wall map compatibility: {compatible} ({compatible_reason})")
    if not compatible:
        raise SystemExit(1)
    print("Plot convention: x-axis=column c, y-axis=row r, row 0 is at the top, points plotted as (c, r).")

    rows, cols, _ = wall_map.shape
    cases = [
        ((0, 0, 0), (min(rows - 1, 2), min(cols - 1, 7))),
        ((min(rows - 1, 3), min(cols - 1, 10), 0), (min(rows - 1, 10), min(cols - 1, 15))),
        ((min(rows - 1, 4), 0, 0), (min(rows - 1, 14), min(cols - 1, 19))),
    ]

    plots = []
    all_ok = run_smoke_checks(wall_map, cases)
    for idx, (start, goal) in enumerate(cases):
        seed = 100 + idx
        result, ok, _ = print_case_report(idx, wall_map, map_source, map_factory, start, goal, seed=seed)
        all_ok = all_ok and ok
        if result["success"]:
            output_path = Path(__file__).with_name(f"rrt_demo_case_{idx}.png")
            plot_path = plot_result(wall_map, result, start, goal, output_path, idx, map_source, map_factory, seed)
            if plot_path is not None:
                plots.append(plot_path)

    print("\nSummary:")
    print(f"map source actually used: {map_source} ({map_factory})")
    print(f"path/debug validation passed for all cases: {all_ok}")
    print(f"plot files saved: {[str(path) for path in plots] if plots else 'none'}")
    print("axes match simulator convention: x=column c, y=row r, row 0 at top")
    print("debug actions are replayed against path_with_theta in the smoke checks and per-case verification")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
