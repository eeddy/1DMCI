import pygame
import random
import time
import socket
import sys # Import sys for graceful exit
import pickle
import math # Import math for trigonometry functions

# Make sure that people can successfully stay at rest!
# 2 Coaching, 3 Evaluation 

MOUSE = False
EVALUATION = True
SPEED = 25

# --- Configuration ---
WINDOW_WIDTH = 850
WINDOW_HEIGHT = 200
TARGET_COUNT = 10 # Number of equally-sized targets
TARGET_WIDTH_RATIO = 0.07 # Width of each target relative to window width
TARGET_HEIGHT_RATIO = 0.5 # Height of each target relative to window height
CURSOR_RADIUS = 10 # Radius of the red cursor circle (Moved up)
TARGET_SPACING = 2 * CURSOR_RADIUS # Pixels of space between targets (Updated to cursor width)
# CURSOR_SPEED = 5 # Pixels per frame - now determined by UDP input 'speed'
DWELL_TIME_SECONDS = 0.5 # Time cursor must stay on target to acquire it
FPS = 60 # Frames per second

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 12346

# Colors (RGB tuples)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TARGET_COLOR_DEFAULT = (200, 200, 200) # Light gray
TARGET_COLOR_HIGHLIGHT = (100, 149, 237) # Cornflower blue
CURSOR_COLOR = (255, 0, 0) # Red
DWELL_FILL_COLOR = (0, 200, 0) # Green for dwell fill

log = {
    'timestamp': [],
    'target': [],
    'cursor': [],
    'trial': [],
}

# --- Game State Variables ---
cursor_x = WINDOW_WIDTH / 2 # Initial cursor position (center)
current_target_index = -1 # Index of the target currently to be acquired
target_acquisition_start_time = None # Time when cursor entered target
game_running = True
active_targets_indices = [] # List of indices of targets remaining in the current trial
trial_number = 0 # Counter for the current trial

# --- Pygame Initialization ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("1D Fitts's Law Test")
clock = pygame.time.Clock()

# --- UDP Socket Setup ---
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False) # Set socket to non-blocking mode
    print(f"UDP Listener: Listening on {UDP_IP}:{UDP_PORT}")
except socket.error as e:
    print(f"Error setting up UDP socket: {e}")
    print("Exiting application.")
    pygame.quit()
    sys.exit()

# --- Utility Functions ---
def get_target_rect(index):
    """Calculates the pygame.Rect object for a target rectangle."""
    # Calculate target dimensions
    target_w = WINDOW_WIDTH * TARGET_WIDTH_RATIO
    target_h = WINDOW_HEIGHT * TARGET_HEIGHT_RATIO

    # Calculate total width occupied by targets and their spacing
    total_occupied_width = (TARGET_COUNT * target_w) + ((TARGET_COUNT - 1) * TARGET_SPACING)
    
    # Calculate starting X to center the row of targets with spacing
    start_x = (WINDOW_WIDTH - total_occupied_width) / 2
    
    # Calculate Y to center targets vertically
    start_y = (WINDOW_HEIGHT - target_h) / 2

    x1 = start_x + index * (target_w + TARGET_SPACING) # Account for spacing
    y1 = start_y
    
    # Pygame Rect takes (left, top, width, height)
    return pygame.Rect(x1, y1, target_w, target_h)

def regenerate_target():
    """
    Randomly selects a new target index from the active_targets_indices list,
    ensuring it's not the one just acquired if there are other options.
    """
    global current_target_index, target_acquisition_start_time

    old_target_index = current_target_index # Store the index of the target just acquired (if any)

    # If there's more than one active target left, pick a new one that isn't the old one
    if len(active_targets_indices) > 1:
        possible_next_targets = [idx for idx in active_targets_indices if idx != old_target_index]
        current_target_index = random.choice(possible_next_targets)
    elif active_targets_indices: # If only one target left, it must be that one
        current_target_index = active_targets_indices[0]
    else: # This case should ideally be handled by reset_trial before calling regenerate_target
        current_target_index = -1 # No target to select

    target_acquisition_start_time = None # Reset dwell timer
    if current_target_index != -1:
        print(f"New target at index: {current_target_index}. Remaining targets: {len(active_targets_indices)}")

def reset_trial():
    """Resets the game state to start a new trial with all targets active."""
    global active_targets_indices, cursor_x, trial_number

    # Pause in practice mode (but skip before first trial)
    if EVALUATION and trial_number % 3 == 0 and trial_number != 0 and trial_number != 15:
        print("Press ENTER to begin the next practice trial...")
        waiting = True
        while waiting:
            try:
                sock.recvfrom(1024)
            except:
                pass 

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False

            # Keep display responsive during pause
            # draw_elements()
            clock.tick(FPS)

    if (trial_number == 5 and not EVALUATION) or (trial_number == 15 and EVALUATION):
        pygame.quit()
        sys.exit() # Ensure the program exits gracefully after logging
    trial_number += 1
    print(f"\n--- Starting Trial {trial_number} ---")
    active_targets_indices = list(range(TARGET_COUNT)) # Reset all targets to active
    cursor_x = WINDOW_WIDTH / 2 # Reset cursor to center
    regenerate_target() # Pick the first target for the new trial

# --- Drawing Function ---
def draw_elements():
    """Draws all targets and the cursor on the screen, including dwell time progress."""
    screen.fill(WHITE) # Fill background with white

    # Draw only active targets
    for i in active_targets_indices: # Loop only through active_targets_indices
        target_rect = get_target_rect(i)
        
        # Highlight only the current target to be acquired
        color = TARGET_COLOR_HIGHLIGHT if i == current_target_index else TARGET_COLOR_DEFAULT
        
        pygame.draw.rect(screen, color, target_rect)
        pygame.draw.rect(screen, BLACK, target_rect, 1) # Black border

    # Draw cursor
    cursor_center = (int(cursor_x), int(WINDOW_HEIGHT / 2))
    pygame.draw.circle(screen, CURSOR_COLOR, cursor_center, CURSOR_RADIUS)
    pygame.draw.circle(screen, BLACK, cursor_center, CURSOR_RADIUS, 1) # Black border

    font = pygame.font.SysFont(None, 36)
    if EVALUATION:
        text = str(trial_number) + '/15'
    else:
        text = str(trial_number) + '/5'
    text_surface = font.render(text, True, (0, 0, 0))
    text_rect = text_surface.get_rect()
    text_rect.topright = (WINDOW_WIDTH - 10, 10)
    screen.blit(text_surface, text_rect)

    # Draw dwell time progress if on target
    if current_target_index != -1:
        target_rect = get_target_rect(current_target_index)
        is_on_target = (cursor_x + CURSOR_RADIUS > target_rect.left and
                        cursor_x - CURSOR_RADIUS < target_rect.right)

        if is_on_target and target_acquisition_start_time is not None:
            time_on_target = time.time() - target_acquisition_start_time
            progress = min(1.0, time_on_target / DWELL_TIME_SECONDS) # Progress from 0 to 1

            # Calculate the angle for the arc
            # Arc starts from the right (0 radians) and goes clockwise
            # We want to fill clockwise from the top (90 degrees or pi/2 radians)
            # A full circle is 2*pi radians (360 degrees)
            # So, progress * 2 * pi gives the angle
            start_angle = math.radians(90) # Start from the top
            end_angle = start_angle + (2 * math.pi * progress)

            # Draw the filling arc
            # The rect for arc should be a square that bounds the circle
            arc_rect = pygame.Rect(cursor_center[0] - CURSOR_RADIUS,
                                   cursor_center[1] - CURSOR_RADIUS,
                                   CURSOR_RADIUS * 2,
                                   CURSOR_RADIUS * 2)

            if progress > 0:
                # Calculate points for a pie slice
                points = [cursor_center]
                # Convert to radians
                start_rad = math.radians(-90) # Top
                end_rad = math.radians(-90 + (progress * 360))

                # Number of segments for the arc to make it smooth
                num_segments = 30 
                for i in range(num_segments + 1):
                    angle = start_rad + (end_rad - start_rad) * i / num_segments
                    x = cursor_center[0] + CURSOR_RADIUS * math.cos(angle)
                    y = cursor_center[1] + CURSOR_RADIUS * math.sin(angle)
                    points.append((x, y))
                
                pygame.draw.polygon(screen, DWELL_FILL_COLOR, points)


    pygame.display.flip() # Update the full display Surface to the screen

# --- Game Loop ---
# Initialize first trial
reset_trial() # This will set up active_targets_indices and pick the first target
draw_elements()

while game_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False
        elif event.type == pygame.KEYDOWN: # Check for KEYDOWN events
            if event.key == pygame.K_ESCAPE: # Allow 'Esc' to quit
                game_running = False
        
    # --- UDP Input Handling ---
    if not MOUSE: # If not using mouse, listen for UDP input
        try:
            data_bytes, _ = sock.recvfrom(1024) # Non-blocking receive
            data_str = data_bytes.decode("utf-8").strip()
            
            # Parse the incoming data
            parts = data_str.split(' ')
            if len(parts) == 2:
                input_class = float(parts[0])
                speed_val = float(parts[1])
                
                speed_val = max(0.0, (speed_val - 5) / 15) # 5 is 3 stds of NM and 15 is about 50 percentile of active contractions
                speed_val = min(1.0, speed_val) ** 2
                
                # Apply movement based on input_class and normalized speed
                # The multiplier (e.g., 20) controls how sensitive the cursor is to the normalized speed
                if input_class == 1: # Left
                    cursor_x -= speed_val * SPEED
                elif input_class == 2: # Right
                    cursor_x += speed_val * SPEED
            else:
                print(f"UDP Listener: Unexpected data format: '{data_str}'")

        except BlockingIOError:
            # No data available, continue game loop without waiting
            pass
        except Exception as e:
            print(f"UDP Listener: An error occurred: {e}")
    else:
        pygame.mouse.set_visible(False) # Hide mouse cursor if using mouse input
        cursor_x = pygame.mouse.get_pos()[0] # Use mouse position for cursor X

    # --- Cursor Bounds Check ---
    first_target_rect = get_target_rect(0)
    last_target_rect = get_target_rect(TARGET_COUNT - 1)
    
    min_x = first_target_rect.left + CURSOR_RADIUS
    max_x = last_target_rect.right - CURSOR_RADIUS
    
    cursor_x = max(min_x, min(max_x, cursor_x))

    # --- Target Acquisition Logic ---
    if current_target_index != -1:
        target_rect = get_target_rect(current_target_index)

        is_on_target = (cursor_x + CURSOR_RADIUS > target_rect.left and
                        cursor_x - CURSOR_RADIUS < target_rect.right)

        current_time = time.time()
        if is_on_target:
            if target_acquisition_start_time is None:
                target_acquisition_start_time = current_time
                # print("Cursor entered target.")
            elif (current_time - target_acquisition_start_time) >= DWELL_TIME_SECONDS:
                print(f"Target acquired! Dwell time: {DWELL_TIME_SECONDS}s. Acquired target index: {current_target_index}")
                
                # Remove the acquired target from the active list
                if current_target_index in active_targets_indices:
                    active_targets_indices.remove(current_target_index)
                
                # Check if all targets have been acquired for the current trial
                if not active_targets_indices:
                    print("All targets acquired! Starting new trial.")
                    reset_trial() # Reset game for a new trial
                else:
                    regenerate_target() # Pick next target from remaining active ones
                
        else:
            # Cursor moved off target, reset timer
            if target_acquisition_start_time is not None:
                # print("Cursor left target. Resetting timer.")
                target_acquisition_start_time = None

    log['timestamp'].append(time.time())
    log['target'].append(current_target_index)
    log['cursor'].append(cursor_x)
    log['trial'].append(trial_number)
    
    draw_elements() # Redraw after movement and state update
    clock.tick(FPS) # Control frame rate

pygame.quit()
print("Application closed.")