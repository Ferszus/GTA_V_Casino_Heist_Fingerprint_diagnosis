import numpy as np
import cv2
from mss import mss
import pygetwindow as gw
import os

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
    'left': second_monitor_window.left // 2,
    'top': int(second_monitor_window.height // 3.85),
    'width': second_monitor_window.width // 4,
    'height': int(second_monitor_window.height // 1.75)
}

# Inicjalizacja detektora SIFT
sift = cv2.SIFT_create()
template_paths = [
            os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "11", "Screenshot_1.crop.png"),
            os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "1", "Screenshot_1.crop.png"),
            os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "111", "Screenshot_1.crop.png"),
            os.path.join(os.getcwd(), "GTA_V_Casino_Heist_Fingerprint_diagnosis", "images", "1111", "Screenshot_1.crop.png")
        ]
# Przechwyć obraz z drugiego monitora
with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        img = np.array(screenShot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the templates
        templates = [cv2.imread(path, 0) for path in template_paths]

        # Wyszukaj keypoints i deskryptory dla obrazu z drugiego monitora
        kp1, des1 = sift.detectAndCompute(gray, None)

        # Przeprowadź dopasowanie dla każdego szablonu
        for template in templates:
            # Resize template for faster matching
            template_resized = cv2.resize(template, None, fx=0.5, fy=0.5)

            kp2, des2 = sift.detectAndCompute(template_resized, None)

            # Użyj metody BFMatcher do dopasowania deskryptorów
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Zastosuj test stosunku dla dopasowań
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Wyznacz macierz transformacji perspektywicznej
                M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                # Zastosuj transformację perspektywiczną do rogu szablonu
                h, w = template_resized.shape
                corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, M)

                # Narysuj prostokąt wokół dopasowanego obszaru
                img = cv2.polylines(img, [np.int32(transformed_corners)], True, (0, 255, 0), 2)

        # Wyświetl wynik
        cv2.imshow('test', img)
        if cv2.waitKey(33) & 0xFF in (
                ord('q'),
                27,
        ):
            break
