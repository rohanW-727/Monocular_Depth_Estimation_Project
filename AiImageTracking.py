import cv2
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression
from aiModel import preprocess_predict_with_overlay

# Checkerboard Calibration 
image_folder = os.path.expanduser("path/to/Checkerboard_Captures")
checkboard_dim = (8, 6)
square_size_mm = 15.0

objp = np.zeros((checkboard_dim[0] * checkboard_dim[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkboard_dim[0], 0:checkboard_dim[1]].T.reshape(-1, 2)
objp *= square_size_mm

objpoints = []
imgpoints = []
first_corners = None

image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
if not image_paths:
    print("No calibration images found."); exit()

for path in image_paths:
    img = cv2.imread(path)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, checkboard_dim, None)
    if found: 
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        if first_corners is None: 
            first_corners = corners_refined

if not objpoints:
    print("No checkerboards detected."); exit()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

if first_corners is not None:
    undistorted = cv2.undistortPoints(first_corners, camera_matrix, dist_coeffs, P=camera_matrix)
    pt1, pt2 = undistorted[0][0], undistorted[1][0]
    d_pix = np.linalg.norm(pt2 - pt1)
else:
    print("No corners available for computing d_pix."); exit()

# Constants 
fx = camera_matrix[0, 0]
sensor_width = 3.6  # mm
fx_mm = fx * (sensor_width / 256)
d_t = [180,170,160,150,140,130,50,20]  # Heights in mm
pdp_values = {}

# Get Predicted Length Using AI Model 
overlay, pred, residual = preprocess_predict_with_overlay('/Users/rohanwadhwa/Desktop/objectImage.heic', 50.0)

# Convert RGB -> BGR for OpenCV
bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

# Show the image
cv2.imshow("Prediction Overlay", bgr_overlay)

# Wait for a key press (0 = wait indefinitely)
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()

R = pred                  # Predicted real-world length in mm
K = 2.54 / 96             # Pixel-to-mm scaling constant

# Compute PDP Values
for i in d_t:
    pdp = R / (K * i * d_pix)
    pdp_values[i] = float(pdp)

# Regression Fit for a1 and a2
reference_height = 180
pdp0 = pdp_values[reference_height]

X = [[1/z, 1/(z**2)] for z in pdp_values]
y = list(pdp_values.values())

model = LinearRegression()
model.fit(X, y)

a1, a2 = model.coef_

a1_dict = {z: a1 for z in pdp_values}
a2_dict = {z: a2 for z in pdp_values}

# Results
print("\n=== GLOBAL MODEL CONSTANTS ===")
print(f"Predicted R Value: {R}")
print(f"PDP₀ (intercept): {pdp0:.6f}")
print(f"a1 (constant):   {a1:.6f}")
print(f"a2 (constant):   {a2:.6f}")

print("\na1_dict = {")
for z in sorted(a1_dict): print(f"    {z}: {a1_dict[z]:.6f},")
print("}")

print("\na2_dict = {")
for z in sorted(a2_dict): print(f"    {z}: {a2_dict[z]:.6f},")
print("}")
