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

# ===== STEP 1: COARSE =====
print("Running coarse autofocus...")
coarse_result = af.coarse_focus()

if not coarse_result.success:
    print("❌ Coarse failed")
    exit()

# ===== STEP 2: FINE TRACKING =====
print("Running fine autofocus tracking...")
af.track(stop_after_seconds=60)

print("Autofocus session complete")
