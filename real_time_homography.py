import cv2
import numpy as np

def detect_and_match_features(frame, template, camera_matrix, dist_coeffs):
    # Undistort the frame
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Convert images to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray_template, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_frame, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    matched_img = cv2.drawMatches(template, keypoints1, frame, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Compute homography if sufficient matches are found
    if len(matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            height, width = template.shape[:2]
            corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3)

    return matched_img, frame

if __name__ == "__main__":
    # Load calibration data
    camera_matrix = np.load("calibration/camera_matrix.npy")
    dist_coeffs = np.load("calibration/dist_coeffs.npy")

    # Load template image
    template_path = "input_images/template.jpg"
    template = cv2.imread(template_path)

    if template is None:
        print("Error: Template image not found!")
        exit()

    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam!")
        exit()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and match features in real-time
        matched_img, output_frame = detect_and_match_features(frame, template, camera_matrix, dist_coeffs)

        # Display results
        cv2.imshow("Matched Features", matched_img)
        cv2.imshow("Output", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

