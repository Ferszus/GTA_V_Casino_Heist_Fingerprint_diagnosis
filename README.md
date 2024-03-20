# Easy CV2 GTA V fingerprint recognition
## NOT WORKING !
![GTAVlogo](./Grand-Theft-Auto-V-GTA-5-Logo.png)
## Additional info
You can add your own photos to the /images/  *"name"*  
folder, and then train program to recognize your face, and also change the confidence, when the text should appear in the 33 line od the `main.py` file
### No camera feed?
Try changing line number 19 in `main.py` file to another number
`camera = cv2.VideoCapture( number)`  
By the default the number is 0 :)
## Libraries required:
pip install opencv-python
pip install opencv-contrib-python
pip install pickle5
pip install numpy

## USAGE
To use this program, firstly you need to train it, by running file `faces-train.py`  
After that, you can run `main.py`, and the camera feed will appear.  
To close the camera feed, press `Q` on the keyboard.
