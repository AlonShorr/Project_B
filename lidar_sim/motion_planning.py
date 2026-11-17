from astar_planner import plan_astar


def plan_path(start, goal, wall_map):
    """
    Plan a path from start to goal using A*.

    start: (r, c) or (r, c, theta_idx)
    goal:  (r, c) target grid cell

    Returns a list of (r, c) cells from start to goal inclusive,
    or an empty list if no path is found.
    """
    start_cell = (int(start[0]), int(start[1]))

    result = plan_astar(wall_map=wall_map, start=start, goal=goal)

    if result["success"]:
        print(
            f"A* result: success, "
            f"nodes_expanded={result['debug'].get('nodes_expanded')}, "
            f"path_length={len(result['path'])}, "
            f"path_cost={result['debug'].get('path_cost'):.2f}"
        )
        return [start_cell] + result["path"]

    print("A* failed:", result["debug"]["reason"])
    return []
