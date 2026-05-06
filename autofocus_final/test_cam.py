from autofocus_final.stage_iface import SerialStage
import time

# ====== SET THIS ======
PORT = "/dev/ttyACM0"

# ====== INIT ======
stage = SerialStage(port=PORT)

# ====== TEST STAGE ======
print("Testing stage movement...")

stage.move_z(100)
time.sleep(1)

stage.move_z(-100)
time.sleep(1)

print("Stage movement OK")

# ====== TEST CAMERA ======
def grab_frame():
    import cv2
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

img = grab_frame()

if img is None:
    print("❌ Camera failed")
else:
    print("Camera working, shape:", img.shape)
