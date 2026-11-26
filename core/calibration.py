import cv2
import numpy as np

# 10 x 7 squares = 9 x 6 INNER corners
PATTERN = (9, 6)
SQUARE_SIZE = 0.025  # square width in metres from chessboard
MIN_VIEWS = 12       # aim for 15-20 varied successful views

# The board is flat: all object-point Z coordinates are 0
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

def main():
    object_points = []
    image_points = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam: try camera index 1")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001)
    image_size = None
    print("SPACE = save a view with all corners found; Q = calibrate")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)
        found, corners = cv2.findChessboardCorners(gray, PATTERN)
        display = frame.copy()
        if found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11),
                                    (-1, -1), criteria)
            cv2.drawChessboardCorners(display, PATTERN, refined, found)
        cv2.putText(display, f"Saved: {len(image_points)} | SPACE / Q",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2)
        cv2.imshow("Calibration chessboard", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            object_points.append(objp.copy())
            image_points.append(refined.copy())
            print(f"Saved view {len(image_points)}")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    if image_size is None or len(image_points) < MIN_VIEWS:
        raise RuntimeError(f"Need at least {MIN_VIEWS} distinct valid views. Try again.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)

    #if one view is too big, the image is deleted and recalibrated
    #therefore rms should be ideally under 0.5px
    for i, (op,ip) in enumerate(zip(object_points, image_points)):
        proj, _= cv2.projectPoints(op,rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(ip.reshape(-1,2),proj.reshape(-1,2), cv2.NORM_L2)/np.sqrt(len(proj))
        print(f"view {i+1}: {err:.3f}px")
    
    print("Camera matrix K:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())
    print(f"RMS reprojection error: {rms:.3f} pixels")
    np.savez("camera_calibration.npz",
            camera_matrix=camera_matrix, 
            dist_coeffs=dist_coeffs,
            image_size=np.array(image_size), 
            rms = rms, 
            num_views = len(image_points))
    print("Saved camera_calibration.npz")

if __name__ == "__main__":
    main()