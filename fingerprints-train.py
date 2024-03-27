import numpy as np
import cv2
from mss import mss
from PIL import ImageGrab
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
    'left': second_monitor_window.left//2,
    'top': second_monitor_window.top//2,
    'width': second_monitor_window.width//2,
    'height': second_monitor_window.height//2
}

# Przechwyć obraz z drugiego monitora
with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        img = np.array(screenShot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = screenShot
        img_rbg = screenShot
        template = cv2.imread(os.getcwd() + r"\GTA_V_Casino_Heist_Fingerprint_diagnosis\images\1\Screenshot_1.crop.png", 0)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6

        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h),
                (0, 0, 255), 2)



        cv2.imshow('test', img)
        if cv2.waitKey(33) & 0xFF in (
            ord('q'), 
            27, 
        ):
            break
