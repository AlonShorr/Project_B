import pygame
import numpy as np
import math

from config import (
    MAP_WIDTH, MAP_HEIGHT, SIDEBAR_WIDTH, WINDOW_SIZE,
    GRID_SIZE, MAP_ROWS, MAP_COLS,
    SENSOR_ANGLES, DIR_TO_ANGLE,
    WALL_UP, WALL_DOWN, WALL_LEFT, WALL_RIGHT,
    LOCALIZATION_THRESHOLD, IS_LOST_THRESHOLD, AUTO_STEP_DELAY,
    LOCALIZATION_AMBIGUITY_RATIO,
    ENABLE_SENSOR_NOISE, ENABLE_MOTION_NOISE, RANDOM_SEED,
    FORWARD_TRANSITION_BIN_HALF_WIDTH_MM, FORWARD_CONTROL_TOLERANCE_MM, FORWARD_SIGMA_MM,
    TURN_TRANSITION_BIN_HALF_WIDTH_DEG, TURN_CONTROL_TOLERANCE_DEG, TURN_SIGMA_DEG,
    BLACK, WHITE, BLUE, GREEN, RED, GRAY, CYAN, YELLOW, ORANGE, SIDEBAR_BG,
)
from sim_robot import Robot
from localization import (
    initialize_belief,
    precompute_all_orientations,
    update_probability,
    predict_motion,
    predict_action,
    get_best_estimate,
    collapse_position_belief,
    dominant_position_modes,
)
from active_localization import select_best_action, apply_action_to_robot
from motion_planning import plan_path
from map_builder import toggle_wall_click, dump_selected_cells
from map_generator import get_maps


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Lidar Sim: WASD=Move, QE=Rotate, R=Auto, B=Builder, P=Analysis")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    small_font = pygame.font.SysFont("Arial", 14)
    sidebar_font = pygame.font.SysFont("Arial", 16)
    tiny_font = pygame.font.SysFont("Arial", 12)

    maps = get_maps()
    current_map_index = 0
    current_map = maps[current_map_index]
    wall_map = np.array(current_map["wall_map"], copy=True)
    expected_data = precompute_all_orientations(wall_map)

    rng = np.random.default_rng(RANDOM_SEED)
    prob_matrix = initialize_belief(MAP_ROWS, MAP_COLS)

    robot = Robot(*current_map["start"])

    analysis_mode = False
    selected_cells = set()
    building_mode = False

    AUTO_IDLE, AUTO_LOCALIZING, AUTO_PLANNING, AUTO_MOVING, AUTO_DONE = 0, 1, 2, 3, 4
    AUTO_STATE_LABELS = {0: "IDLE", 1: "LOCALIZING", 2: "PLANNING", 3: "MOVING", 4: "DONE"}
    auto_state = AUTO_IDLE
    planned_path = []
    active_loc_action = None
    auto_step_delay = AUTO_STEP_DELAY
    last_auto_step_ms = 0
    goal = current_map["goal"]

    def select_map(index):
        nonlocal current_map_index, current_map, wall_map, expected_data
        nonlocal prob_matrix, goal
        if not 0 <= index < len(maps):
            return
        current_map_index = index
        current_map = maps[index]
        wall_map = np.array(current_map["wall_map"], copy=True)
        goal = current_map["goal"]
        robot.move_to(*current_map["start"])
        reset_auto()
        prob_matrix = initialize_belief(MAP_ROWS, MAP_COLS)
        expected_data = precompute_all_orientations(wall_map)
        print(f"Selected map F{index + 1}: {current_map['name']} ({current_map['difficulty']})")

    def reset_auto(state=AUTO_IDLE):
        """
        Functions that resets the state and other variables regarding
        localization and planning. 
        """
        nonlocal auto_state, active_loc_action, planned_path, last_auto_step_ms
        
        auto_state = state          # set state
        active_loc_action = None    # 
        planned_path = []
        last_auto_step_ms = 0

    def start_auto():
        reset_auto(AUTO_LOCALIZING)
        print("--- AUTO MODE STARTED: Localizing... ---")

    def stop_auto():
        reset_auto(AUTO_IDLE)
        print("--- AUTO MODE STOPPED ---")

    map_keys = (
        pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5,
        pygame.K_F6, pygame.K_F7, pygame.K_F8, pygame.K_F9, pygame.K_F10,
    )

    running = True
    while running:
        moved = False
        dr, dc = 0, 0
        commanded_action = None
        commanded_noisy = False

        def handle_keydown(key):
            nonlocal analysis_mode, building_mode, expected_data, prob_matrix
            nonlocal auto_step_delay
            nonlocal dr, dc, moved, commanded_action

            def exit_builder():
                nonlocal building_mode, expected_data, prob_matrix
                if not building_mode:
                    return
                building_mode = False
                screen.fill(BLACK)
                txt = font.render("Recomputing Map Data... Please Wait...", True, WHITE)
                screen.blit(txt, (MAP_WIDTH // 2 - 150, MAP_HEIGHT // 2))
                pygame.display.flip()
                expected_data = precompute_all_orientations(wall_map)
                prob_matrix = initialize_belief(MAP_ROWS, MAP_COLS)

            if key in map_keys:
                select_map(map_keys.index(key))
                return

            if key == pygame.K_b:
                if auto_state not in (AUTO_IDLE, AUTO_DONE):
                    reset_auto()
                if analysis_mode:
                    selected_cells.clear()
                    analysis_mode = False
                if building_mode:
                    exit_builder()
                else:
                    building_mode = True
                return

            if key == pygame.K_p:
                if auto_state not in (AUTO_IDLE, AUTO_DONE):
                    reset_auto()
                if building_mode:
                    exit_builder()
                if analysis_mode:
                    print("\n--- EXITING ANALYSIS MODE ---")
                    if selected_cells:
                        dump_selected_cells(expected_data, selected_cells)
                    else:
                        print("No cells selected.")
                    selected_cells.clear()
                    analysis_mode = False
                else:
                    analysis_mode = True
                    print("\n--- ANALYSIS MODE STARTED ---")
                    print("1. Click cells on the map to mark them (Cyan).")
                    print("2. Press 'P' again to dump their distance data to console.")
                return

            if key == pygame.K_r:
                if auto_state in (AUTO_IDLE, AUTO_DONE):
                    if building_mode:
                        exit_builder()
                    if analysis_mode:
                        selected_cells.clear()
                        analysis_mode = False
                    start_auto()
                else:
                    stop_auto()
                return

            if key == pygame.K_LEFTBRACKET:
                auto_step_delay = max(50, auto_step_delay - 50)
                return
            if key == pygame.K_RIGHTBRACKET:
                auto_step_delay = min(2000, auto_step_delay + 50)
                return

            if not building_mode and not analysis_mode and auto_state in (AUTO_IDLE, AUTO_DONE):
                if auto_state == AUTO_DONE:
                    reset_auto()
                if key == pygame.K_w:
                    dr, dc = -1, 0
                elif key == pygame.K_s:
                    dr, dc = 1, 0
                elif key == pygame.K_a:
                    dr, dc = 0, -1
                elif key == pygame.K_d:
                    dr, dc = 0, 1
                elif key == pygame.K_q:
                    robot.rotate(-1)
                    commanded_action = "TURN_LEFT"
                elif key == pygame.K_e:
                    robot.rotate(1)
                    commanded_action = "TURN_RIGHT"
                elif key == pygame.K_1:
                    robot.toggle_sensor(0)
                elif key == pygame.K_2:
                    robot.toggle_sensor(1)
                elif key == pygame.K_3:
                    robot.toggle_sensor(2)
                elif key == pygame.K_4:
                    robot.toggle_sensor(3)
                elif key == pygame.K_5:
                    robot.toggle_sensor(4)

                if dr != 0 or dc != 0:
                    if robot.move_rel(dr, dc, wall_map):
                        moved = True

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if mx < MAP_WIDTH:
                    if analysis_mode:
                        c, r = mx // GRID_SIZE, my // GRID_SIZE
                        if 0 <= c < MAP_COLS and 0 <= r < MAP_ROWS:
                            coord = (r, c)
                            if coord in selected_cells:
                                selected_cells.remove(coord)
                            else:
                                selected_cells.add(coord)

                    elif building_mode:
                        toggle_wall_click(wall_map, mx, my)

                    else:
                        c, r = mx // GRID_SIZE, my // GRID_SIZE
                        if 0 <= c < MAP_COLS and 0 <= r < MAP_ROWS:
                            if event.button == 3:  # Right-click: set goal
                                goal = (r, c)
                                planned_path = []
                                if auto_state == AUTO_MOVING:
                                    auto_state = AUTO_PLANNING
                                print(f"Goal set to {goal}")
                            else:  # Left-click: teleport robot
                                robot.move_to(r, c)
                                prob_matrix = initialize_belief(MAP_ROWS, MAP_COLS)

            elif event.type == pygame.KEYDOWN:
                handle_keydown(event.key)

        # --- Simulation logic ---
        if not building_mode:
            eff_dr, eff_dc, eff_moved = 0, 0, False
            eff_action = commanded_action
            eff_action_noisy = commanded_noisy
            now = pygame.time.get_ticks()
            auto_step_ready = (now - last_auto_step_ms) >= auto_step_delay

            if auto_state == AUTO_LOCALIZING and auto_step_ready:                    # robot in localization mode and ready for next step 
                if active_loc_action is None:                                           # if robot doesn't know where it is
                    result = select_best_action(                                            # select the action which will give it the most info
                        prob_matrix, expected_data, wall_map, robot.angle_index             # about where it is.
                    )
                    active_loc_action = result["action"]
                    if active_loc_action is not None:
                        print("Active loc:", result["reason"])
                if active_loc_action is None:
                    auto_state = AUTO_PLANNING  # belief already localized
                else:
                    eff_dr, eff_dc, eff_action = apply_action_to_robot(
                        robot, active_loc_action, wall_map, rng, ENABLE_MOTION_NOISE
                    )
                    eff_action_noisy = ENABLE_MOTION_NOISE
                    eff_moved = (eff_dr != 0 or eff_dc != 0)
                    if robot.angle_index == active_loc_action.heading:
                        active_loc_action = None  # action complete, re-select next step
                    last_auto_step_ms = now

            elif auto_state == AUTO_PLANNING:                       
                est_r, est_c, est_theta, _ = get_best_estimate(prob_matrix)
                path = plan_path((est_r, est_c, est_theta), goal, wall_map)
                if path and len(path) > 1:
                    planned_path = list(path[1:])
                    auto_state = AUTO_MOVING
                    print(f"Path found: {len(planned_path)} steps to goal {goal}")
                elif auto_state != AUTO_IDLE:
                    print("A* returned no path. Stopping auto mode.")
                    reset_auto()

            elif auto_state == AUTO_MOVING and auto_step_ready:                      # otherwise, if mode is MOVING and ready to do next step                                                
                while planned_path and (robot.r, robot.c) == planned_path[0]:           # pop current move if robot is on it to get next move
                    planned_path.pop(0)

                if not planned_path:                                                    # if no more moves left in plan
                    if (robot.r, robot.c) == goal:                                          # check if reach goal 
                        reset_auto(AUTO_DONE)
                        print("--- GOAL REACHED! ---")
                    else:                                                                   # if didn't reach return to planning
                        auto_state = AUTO_PLANNING
                
                else:                                                                   # there are still some moves left to do in plane
                    target_r, target_c = planned_path[0]                                    # get the move and check if valid
                    step_dr = target_r - robot.r
                    step_dc = target_c - robot.c
                    required_angle = DIR_TO_ANGLE.get((step_dr, step_dc))
                    
                    if required_angle is None:                                              # next step isn't adjacent 
                        print("Next path step is non-adjacent — belief may be wrong. Re-localizing...")
                        reset_auto(AUTO_LOCALIZING)                                             # return to localization and reset belief
                    
                    elif robot.angle_index != required_angle:                               # next move is adjacent and in different angle from current robot orientation
                        delta = (required_angle - robot.angle_index) % 8                        # pick shortest turn 
                        turn_action = "TURN_RIGHT" if delta <= 4 else "TURN_LEFT"
                        if ENABLE_MOTION_NOISE:                                                 # if we enabled motion noise apply it on action
                            robot.apply_action(turn_action, wall_map, rng, noisy=True)          
                        else:                                                                   # otherwise just rotate 
                            robot.rotate(1 if delta <= 4 else -1)                               
                        eff_action = turn_action                                                # update the effective action we are doing,
                        eff_action_noisy = ENABLE_MOTION_NOISE                                  # noise and time
                        last_auto_step_ms = now
                    
                    else:                                                                   # if next move is in our current direction - it's forward
                        old_r, old_c = robot.r, robot.c                                         # update old coords (r,c)
                        if ENABLE_MOTION_NOISE:                                                 # move robot forward and add noise if enabled
                            robot.apply_action("FORWARD", wall_map, rng, noisy=True)            
                        else:
                            robot.move_rel(step_dr, step_dc, wall_map)
                        eff_dr, eff_dc = robot.r - old_r, robot.c - old_c                       # update effective row and column step dr
                        eff_moved = (eff_dr != 0 or eff_dc != 0)                                # and if moved flag it
                        eff_action = "FORWARD"
                        eff_action_noisy = ENABLE_MOTION_NOISE
                        last_auto_step_ms = now
                        if robot.r == target_r and robot.c == target_c:                         # if reached target pos given by planner
                            planned_path.pop(0)                                                     # continue to next move
                        elif not eff_moved:                                                     # otherwise if it didn't move - it got stuck
                            print("Forward move blocked by wall — plan is invalid. Re-localizing...")
                            reset_auto(AUTO_LOCALIZING)                                             # mode: MOVING -> LOCALIZING

            else:  # AUTO_IDLE or AUTO_DONE — manual control
                eff_dr, eff_dc = dr, dc
                eff_moved = moved

            if eff_action in ("FORWARD", "TURN_LEFT", "TURN_RIGHT"):
                prob_matrix = predict_action(
                    prob_matrix, eff_action, wall_map, noisy=eff_action_noisy
                )
            elif eff_moved:
                prob_matrix = predict_motion(prob_matrix, eff_dr, eff_dc, wall_map, noisy=False)

            dists, hit_points = robot.measure(wall_map, add_noise=ENABLE_SENSOR_NOISE)
            prob_matrix = update_probability(prob_matrix, dists, expected_data)

            if auto_state == AUTO_LOCALIZING:
                peak, runner_up = dominant_position_modes(prob_matrix)
                ambiguous = runner_up > peak * LOCALIZATION_AMBIGUITY_RATIO
                if peak >= LOCALIZATION_THRESHOLD and not ambiguous:
                    print(f"Localized! peak={peak:.3f} (runner-up {runner_up:.3f}). "
                          f"Planning path to {goal}...")
                    auto_state = AUTO_PLANNING
                    active_loc_action = None
            elif auto_state == AUTO_MOVING:
                _, _, _, peak = get_best_estimate(prob_matrix)
                if peak < IS_LOST_THRESHOLD:
                    print(f"Lost (peak={peak:.3f}). Re-localizing...")
                    reset_auto(AUTO_LOCALIZING)
        else:
            dists, hit_points = robot.measure(wall_map, add_noise=ENABLE_SENSOR_NOISE)

        # --- Drawing ---
        screen.fill(BLACK)
        position_prob = collapse_position_belief(prob_matrix)

        for r in range(MAP_ROWS):
            for c in range(MAP_COLS):
                rect = (c * GRID_SIZE, r * GRID_SIZE, GRID_SIZE, GRID_SIZE)

                if not building_mode and not analysis_mode:
                    b = min(int(position_prob[r, c] * 5000), 255)
                    if b > 10:
                        s = pygame.Surface((GRID_SIZE, GRID_SIZE))
                        s.set_alpha(150)
                        s.fill((b, 0, 0))
                        screen.blit(s, rect[:2])

                if (r, c) == goal:
                    goal_s = pygame.Surface((GRID_SIZE, GRID_SIZE))
                    goal_s.set_alpha(180)
                    goal_s.fill(YELLOW)
                    screen.blit(goal_s, rect[:2])

                grid_col = (60, 60, 0) if building_mode else (30, 30, 30)
                pygame.draw.rect(screen, grid_col, rect, 1)

                if analysis_mode and (r, c) in selected_cells:
                    pygame.draw.rect(screen, CYAN, rect)

                w_thick = 4 if building_mode else 2
                if wall_map[r, c, WALL_UP] == 0:
                    pygame.draw.line(screen, RED, (c * GRID_SIZE, r * GRID_SIZE), ((c + 1) * GRID_SIZE, r * GRID_SIZE), w_thick)
                if wall_map[r, c, WALL_DOWN] == 0:
                    pygame.draw.line(screen, RED, (c * GRID_SIZE, (r + 1) * GRID_SIZE), ((c + 1) * GRID_SIZE, (r + 1) * GRID_SIZE), w_thick)
                if wall_map[r, c, WALL_LEFT] == 0:
                    pygame.draw.line(screen, RED, (c * GRID_SIZE, r * GRID_SIZE), (c * GRID_SIZE, (r + 1) * GRID_SIZE), w_thick)
                if wall_map[r, c, WALL_RIGHT] == 0:
                    pygame.draw.line(screen, RED, ((c + 1) * GRID_SIZE, r * GRID_SIZE), ((c + 1) * GRID_SIZE, (r + 1) * GRID_SIZE), w_thick)

        if analysis_mode:
            pygame.draw.rect(screen, BLUE, (0, 0, MAP_WIDTH, MAP_HEIGHT), 5)

        pygame.draw.circle(screen, BLUE, (int(robot.true_x), int(robot.true_y)), 10)
        head_rad = math.radians(robot.angle_deg)
        head_x = robot.true_x + math.cos(head_rad) * 15
        head_y = robot.true_y + math.sin(head_rad) * 15
        pygame.draw.line(screen, WHITE, (robot.true_x, robot.true_y), (head_x, head_y), 2)

        for p in hit_points:
            pygame.draw.line(screen, GREEN, (robot.true_x, robot.true_y), p, 1)
            pygame.draw.circle(screen, GREEN, (int(p[0]), int(p[1])), 2)

        if planned_path:
            prev_px, prev_py = int(robot.true_x), int(robot.true_y)
            for wp_r, wp_c in planned_path:
                wp_x = int(wp_c * GRID_SIZE + GRID_SIZE // 2)
                wp_y = int(wp_r * GRID_SIZE + GRID_SIZE // 2)
                pygame.draw.line(screen, ORANGE, (prev_px, prev_py), (wp_x, wp_y), 2)
                pygame.draw.circle(screen, ORANGE, (wp_x, wp_y), 3)
                prev_px, prev_py = wp_x, wp_y

        # --- Sidebar ---
        pygame.draw.rect(screen, SIDEBAR_BG, pygame.Rect(MAP_WIDTH, 0, SIDEBAR_WIDTH, MAP_HEIGHT))
        pygame.draw.line(screen, WHITE, (MAP_WIDTH, 0), (MAP_WIDTH, MAP_HEIGHT), 2)

        x_start = MAP_WIDTH + 10
        y_off = 10
        sidebar_right = MAP_WIDTH + SIDEBAR_WIDTH - 8
        legend_height = (tiny_font.get_linesize() + 2) * 2
        legend_y = MAP_HEIGHT - legend_height - 8
        content_bottom = legend_y - 6
        tiny_step = tiny_font.get_linesize() + 1

        def draw_text(text, text_font, color, gap=2, bottom=content_bottom):
            nonlocal y_off
            line_height = text_font.get_linesize()
            if y_off + line_height <= bottom:
                surface = text_font.render(text, True, color)
                max_width = sidebar_right - x_start
                if surface.get_width() > max_width:
                    clipped = text
                    while clipped and surface.get_width() > max_width:
                        clipped = clipped[:-1]
                        surface = text_font.render(clipped + "...", True, color)
                    if clipped:
                        surface = text_font.render(clipped + "...", True, color)
                screen.blit(surface, (x_start, y_off))
            y_off += line_height + gap

        def draw_section_title(text):
            draw_text(text, sidebar_font, GREEN, gap=3)

        if analysis_mode:
            mode_txt, mode_col = "ANALYSIS MODE", CYAN
        elif building_mode:
            mode_txt, mode_col = "BUILDER MODE", YELLOW
        elif auto_state == AUTO_DONE:
            mode_txt, mode_col = "GOAL REACHED!", GREEN
        elif auto_state != AUTO_IDLE:
            mode_txt, mode_col = f"AUTO: {AUTO_STATE_LABELS[auto_state]}", ORANGE
        else:
            mode_txt, mode_col = "SIMULATION MODE", GREEN

        draw_text(mode_txt, sidebar_font, mode_col, gap=3)
        draw_section_title("MAP")
        draw_text(f"F{current_map_index + 1}: {current_map['name']}", tiny_font, YELLOW)
        draw_text(f"Difficulty: {current_map['difficulty']}", tiny_font, WHITE)
        draw_text(f"Ambiguity: {current_map['ambiguity_score']}", tiny_font, WHITE)
        draw_text(f"Shortest path: {current_map['shortest_path_steps']}", tiny_font, WHITE)
        draw_text("F1-F10: Select Map", tiny_font, WHITE, gap=4)

        draw_section_title("CONTROLS")
        draw_text("B: Toggle Builder", tiny_font, WHITE)
        draw_text("P: Toggle Analysis", tiny_font, WHITE)

        if building_mode:
            draw_text("Click Edges: Toggle Wall", tiny_font, YELLOW, gap=4)
        elif analysis_mode:
            draw_text("Click Cells: Select/Deselect", tiny_font, CYAN)
            draw_text("Press P again to Dump Data", tiny_font, CYAN, gap=4)
        elif auto_state == AUTO_DONE:
            draw_text(f"Goal {goal} reached!", tiny_font, GREEN)
            draw_text("R: Run Auto Again", tiny_font, WHITE, gap=4)
        elif auto_state != AUTO_IDLE:
            draw_text(f"Goal: row={goal[0]} col={goal[1]}", tiny_font, YELLOW)
            draw_text(f"Path: {len(planned_path)} steps left", tiny_font, WHITE)
            draw_text(f"Step delay: {auto_step_delay}ms  [ / ]", tiny_font, WHITE)
            draw_text("R: Stop Auto", tiny_font, ORANGE, gap=4)
        else:
            draw_text("WASD: Move  |  Q/E: Rotate", tiny_font, WHITE)
            draw_text("LClick: Teleport", tiny_font, WHITE)
            draw_text(f"RClick: Set Goal {goal}", tiny_font, YELLOW)
            draw_text(f"Step delay: {auto_step_delay}ms  [ / ]", tiny_font, WHITE)
            draw_text("R: Run Auto", tiny_font, WHITE, gap=4)

        draw_section_title("SENSORS (Keys 1-5)")
        for i, active in enumerate(robot.active_sensors):
            color = GREEN if active else RED
            draw_text(f"#{i+1} ({SENSOR_ANGLES[i]}°): {'ON' if active else 'OFF'}", tiny_font, color)
        y_off += 2

        draw_section_title("NOISE")
        draw_text(f"Sensor noise: {ENABLE_SENSOR_NOISE}", tiny_font, WHITE)
        draw_text(f"Motion noise: {ENABLE_MOTION_NOISE}", tiny_font, WHITE)
        draw_text(f"F bin/sigma: {FORWARD_TRANSITION_BIN_HALF_WIDTH_MM:.1f}/{FORWARD_SIGMA_MM:.1f} mm", tiny_font, WHITE)
        draw_text(f"T bin/sigma: {TURN_TRANSITION_BIN_HALF_WIDTH_DEG:.1f}/{TURN_SIGMA_DEG:.1f} deg", tiny_font, WHITE)
        draw_text(f"Ctrl tol: F {FORWARD_CONTROL_TOLERANCE_MM:.1f}mm, T {TURN_CONTROL_TOLERANCE_DEG:.1f}deg", tiny_font, WHITE)
        y_off += 2

        draw_section_title("DIAGNOSTICS")
        draw_text(f"Heading: {robot.angle_deg}°", tiny_font, WHITE)
        draw_text("Top Probable Locs:", tiny_font, GREEN)

        if not building_mode and not analysis_mode:
            count = 0
            max_rows = max(0, (content_bottom - y_off) // tiny_step)
            for idx in np.argsort(prob_matrix.ravel())[::-1]:
                r_idx, c_idx, theta_idx = np.unravel_index(idx, prob_matrix.shape)
                prob = prob_matrix[r_idx, c_idx, theta_idx]
                if prob < 0.005 or count >= min(8, max_rows):
                    break
                text_str = f"({r_idx}, {c_idx}, {theta_idx}): {prob:.3f}"
                color = WHITE
                if r_idx == robot.r and c_idx == robot.c and theta_idx == robot.angle_index:
                    color = GREEN
                    text_str += " <--"
                draw_text(text_str, tiny_font, color)
                count += 1
        elif analysis_mode:
            draw_text(f"Selected: {len(selected_cells)}", tiny_font, CYAN)
        else:
            draw_text("Paused...", tiny_font, GRAY)

        y_off = legend_y
        draw_text("Map Legend:", tiny_font, GREEN, bottom=MAP_HEIGHT - 4)
        draw_text("Red Intensity = Probability", tiny_font, RED, bottom=MAP_HEIGHT - 4)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
