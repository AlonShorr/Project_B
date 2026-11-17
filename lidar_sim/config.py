# --- Display ---
MAP_WIDTH = 800
MAP_HEIGHT = 600
SIDEBAR_WIDTH = 250
WINDOW_SIZE = (MAP_WIDTH + SIDEBAR_WIDTH, MAP_HEIGHT)

GRID_SIZE = 40  # pixels
MAP_COLS = MAP_WIDTH // GRID_SIZE
MAP_ROWS = MAP_HEIGHT // GRID_SIZE

# --- Physics / Sensor ---
CELL_SIZE_MM = 180.0
PIXELS_PER_MM = GRID_SIZE / CELL_SIZE_MM
MIN_RANGE_MM = 1.0
MAX_RANGE_MM = 1300.0
SENSOR_ACCURACY_MM = 1.0  # Standard deviation for noise (mm)
MIN_RANGE_PX = MIN_RANGE_MM * PIXELS_PER_MM
MAX_RANGE_PX = MAX_RANGE_MM * PIXELS_PER_MM

# 0 is east, angle grows clockwise
SENSOR_ANGLES = [-90, -40.33, 0, 37.22, 90]  # Relative to robot facing
ROTATION_STEP = 45  # Degrees per Q/E press

# --- Discrete stochastic noise model ---
ENABLE_SENSOR_NOISE = True
ENABLE_MOTION_NOISE = True
RANDOM_SEED = 42

# Forward movement model
FORWARD_TARGET_MM = CELL_SIZE_MM
FORWARD_SIGMA_MM = 15.0
FORWARD_BIAS_MM = 0.0
FORWARD_TRANSITION_BIN_HALF_WIDTH_MM = CELL_SIZE_MM / 2.0
FORWARD_CONTROL_TOLERANCE_MM = 25.0
FORWARD_EXTRA_STAY_PROB = 0.02
FORWARD_EXTRA_OVERSHOOT_PROB = 0.01

# Turn model
TURN_TARGET_DEG = ROTATION_STEP
TURN_SIGMA_DEG = 4.0
TURN_BIAS_DEG = 0.0
TURN_TRANSITION_BIN_HALF_WIDTH_DEG = ROTATION_STEP / 2.0
TURN_CONTROL_TOLERANCE_DEG = 7.5
TURN_EXTRA_UNDERTURN_PROB = 0.02
TURN_EXTRA_OVERTURN_PROB = 0.02

# Sensor likelihood model
SENSOR_LIKELIHOOD_SIGMA_MM = 20.0
SENSOR_LIKELIHOOD_FLOOR = 1e-9
SENSOR_LIKELIHOOD_POWER = 1.0

# Optional actual measurement perturbation
SENSOR_MEASUREMENT_SIGMA_MM = SENSOR_ACCURACY_MM

# --- Wall indices (for wall_map array) ---
# wall_map shape: (Rows, Cols, 4). Values: 0=Wall, 1=Open
WALL_UP = 0
WALL_DOWN = 1
WALL_LEFT = 2
WALL_RIGHT = 3

# --- Algorithm thresholds ---
LOCALIZATION_THRESHOLD = 0.25   # Min peak-prob to trust localization
LOCALIZATION_MOVES = 8          # Max random exploratory moves while lost
IS_LOST_THRESHOLD = 0.15        # Re-localize if peak-prob drops below this
# Belief is "localized" only if the strongest competing (far-away) mode is at
# or below this fraction of the peak. Guards against a bimodal belief (two
# equiprobable cells) being trusted as localized, which sends a bad plan to A*.
LOCALIZATION_AMBIGUITY_RATIO = 0.5
AUTO_STEP_DELAY = 200           # ms between auto steps; adjust with [ / ] keys

GOAL = (1, 8)  # Default (row, col) goal

# Maps (dr, dc) movement direction to angle_index (6=N, 4=W,  2=S, 0=E)
DIR_TO_ANGLE = {(-1, 0): 6, (0, -1): 4, (1, 0): 2, (0, 1): 0}

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (50, 50, 50)
CYAN = (0, 255, 255)
DARK_RED = (100, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
SIDEBAR_BG = (30, 30, 30)
