import cv2
import numpy as np
import os

def calibrate_camera(calib_images_dir, chessboard_size=(9, 6)):
    # Prepare object points (e.g., (0,0,0), (1,0,0), ..., (8,5,0))
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Load calibration images
    images = [os.path.join(calib_images_dir, f) for f in os.listdir(calib_images_dir) if f.endswith(('.jpg', '.png'))]
    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display corners
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Calibration Image', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("Camera calibration successful!")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)
        return mtx, dist
    else:
        print("Camera calibration failed!")
        return None, None

if __name__ == "__main__":
    calib_images_dir = "calibration/calibration_images/"
    os.makedirs(calib_images_dir, exist_ok=True)
    print("Place chessboard calibration images in 'calibration/calibration_images/' folder.")
    input("Press Enter after placing the images...")
    camera_matrix, dist_coeffs = calibrate_camera(calib_images_dir)

    if camera_matrix is not None:
        # Save calibration results
        np.save("calibration/camera_matrix.npy", camera_matrix)
        np.save("calibration/dist_coeffs.npy", dist_coeffs)
        print("Calibration data saved in 'calibration/' folder.")

