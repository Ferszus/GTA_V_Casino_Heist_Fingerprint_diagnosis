import numpy as np
import cv2
from mss import mss
import pygetwindow as gw
import os
import time
import subprocess
from screeninfo import get_monitors

positions = []

def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time


### Append image paths to the template
template_paths = []
for folder in range(1, 5):
    for name in range(1, 5):
            path = os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", str(folder), f"Finger{folder}_{name}.png")
            #print(path)
            template_paths.append(path)


# Define scaling factors for multi-scale template matching
scale_factors = [0.8, 0.9, 1, 1.1, 1.2]  # Adjust as needed
threshold = 0.7  # Adjust as needed

# Define colors for different folders
folder_colors = {
    "1": (0, 0, 255),  # Red
    "2": (255, 0, 0),  # Blue
    "3": (0, 255, 0),  # Green
    "4": (0, 255, 255)  # Yellow
}

# Get all open windows
windows = gw.getAllWindows()

# Choose the window on the second monitor (if any)
second_monitor_window = None
for window in windows:
    if window.isMaximized and window.top > 0:
        second_monitor_window = window
        break

# If no window is found on the second monitor, exit the program
if second_monitor_window is None:
    print("No open window found on the second monitor.")
    exit()
def MainProgram():
    # Set capture region based on window coordinates
    mon = {
        'left': positions[0],
        'top': positions[1],
        'width': positions[2],
        'height': positions[3]
    }
    print(mon)
    with mss() as sct:
        prev_time = time.time()
        while True:
            screenShot = sct.grab(mon)
            img = np.array(screenShot)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = screenShot
            img_rbg = screenShot



            templates = [cv2.imread(path, 0) for path in template_paths]

            # Perform multi-scale template matching for each template
            for template_path, template in zip(template_paths, templates):
                folder_name = os.path.basename(os.path.dirname(template_path))
                template_color = folder_colors.get(folder_name, (0, 0, 0))  # Default color is black if folder not found

                found = None
                for scale_factor in scale_factors:
                    resized_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
                    result = cv2.matchTemplate(resized_gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    # If a match is found and exceeds the threshold, store the information
                    if max_val > threshold and (found is None or max_val > found[0]):
                        found = (max_val, max_loc, scale_factor)

                # Extract the best match for the current template
                if found is not None:
                    max_val, max_loc, scale_factor = found
                    w, h = template.shape[::-1]
                    pt = (int(max_loc[0] / scale_factor), int(max_loc[1] / scale_factor))
                    pt_end = (int((max_loc[0] + w) / scale_factor), int((max_loc[1] + h) / scale_factor))

                    # Draw rectangle around the match on the original image with corresponding color
                    cv2.rectangle(img, pt, pt_end, template_color, 2)

            fps, prev_time = calculate_fps(prev_time)
            print(f"FPS: {fps:.2f}", end="\r")
            # Display the result
            cv2.imshow('Fingerprint_CHECKER', img)
            if cv2.waitKey(33) & 0xFF in (
                    ord('q'),
                    27,
            ):
                break


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
def click_and_crop(event, x, y, flags, param):
    global positions
    # Global variables needed to store initial position
    global ref_point, cropping

    # Left mouse button click event handling
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]  # Store initial position
        cropping = True

    # Left mouse button release event handling
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))  # Store end position
        cropping = False
        
        # Draw rectangle on the image
        cv2.rectangle(screen, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Screen", screen)

        # Determine position, width, and height of the rectangle
        x_start, y_start = ref_point[0]
        x_end, y_end = ref_point[1]
        width = abs(x_end - x_start)
        height = abs(y_end - y_start)

        positions = [x_start, y_start, width, height]
        print("Starting Position (x, y):", x_start, ",", y_start)
        print("Width:", width, "pixels")
        print("Height:", height, "pixels")

        # Pass position information through standard output
        print(" ".join(map(str, positions)))

        # End program after displaying information about selected region
        cv2.destroyAllWindows()
        MainProgram()
        exit()

def get_primary_monitor_number():
    monitors = get_monitors()
    for idx, monitor in enumerate(monitors):
        if monitor.x == 0 and monitor.y == 0:
            return idx + 1  # Monitor numbering starts from 1
# Initialize screen capture object
with mss() as sct:
    # Determine screen area to capture
    monitor = sct.monitors[get_primary_monitor_number()]

    # Initial values for global variables
    cropping = False
    ref_point = []

    # Capture screen image
    screen = np.array(sct.grab(monitor))

    # Clone the image for changes
    clone = screen.copy()

    # Create window to display image and define mouse event handling function
    cv2.imshow("Select scanning area", screen)
    cv2.setMouseCallback("Select scanning area", click_and_crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
