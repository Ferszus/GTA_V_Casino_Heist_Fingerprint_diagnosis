import numpy as np
import cv2
from mss import mss
import pygetwindow as gw
import os
import time

def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time

template_paths = [
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "1", "Finger1_1.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "1", "Finger1_2.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "1", "Finger1_3.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "1", "Finger1_4.png"),
    #os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "1", "WholeFinger1.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "2", "Finger2_1.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "2", "Finger2_2.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "2", "Finger2_3.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "2", "Finger2_4.png"),
    #os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "2", "WholeFinger2.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "3", "Finger3_1.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "3", "Finger3_2.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "3", "Finger3_3.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "3", "Finger3_4.png"),
    #os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "3", "WholeFinger3.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "4", "Finger4_1.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "4", "Finger4_2.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "4", "Finger4_3.png"),
    os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "4", "Finger4_4.png"),
    #os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "4", "WholeFinger4.png")
]
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

# Pobierz wszystkie otwarte okna
windows = gw.getAllWindows()

# Wybierz okno, które znajduje się na drugim monitorze (jeśli istnieje)
second_monitor_window = None
for window in windows:
    if window.isMaximized and window.top > 0:
        second_monitor_window = window
        break

# Jeśli nie znaleziono okna na drugim monitorze, zakończ program
if second_monitor_window is None:
    print("Brak otwartego okna na drugim monitorze.")
    exit()

# Ustaw obszar przechwytywania na podstawie współrzędnych okna
mon = {
    'left': int(second_monitor_window.right +  2560/18*4),
    'top': 320,
    'width': int(second_monitor_window.width//4),
    'height': second_monitor_window.height//5*4
}
print(mon)

# Przechwyć obraz z drugiego monitora
with mss() as sct:
    prev_time = time.time()
    while True:
        screenShot = sct.grab(mon)
        img = np.array(screenShot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = screenShot
        img_rbg = screenShot

        # Load the templates


        # for folder in range(1, 5):
        #     for name in range(1, 6):
        #         if(name != 5):
        #             template_paths.append(os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", folder, f"Finger{folder}_{name}.png"))
        #         else:
        #             template_paths.append(os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", folder, f"WholeFinger{folder}.png"))


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

