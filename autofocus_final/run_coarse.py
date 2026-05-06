from autofocus_final.system import AutofocusSystem
from autofocus_final.stage_iface import SerialStage
import cv2

PORT = "/dev/ttyACM0"

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

def grab_frame():
    ret, frame = cap.read()
    return frame if ret else None

# ===== STAGE =====
stage = SerialStage(port=PORT)

# ===== SYSTEM =====
af = AutofocusSystem(
    grab_frame=grab_frame,
    stage=stage
)

# ===== RUN COARSE =====
result = af.coarse_focus()

print("\n=== COARSE RESULT ===")
print("Success:", result.success)
print("Best Z:", result.best_z)
print("Score:", result.best_score)
print("SNR:", result.snr)
print("Prominence:", result.prominence)
