import cv2
import cv2.aruco as aruco
import numpy as np

# Define the ArUco dictionary and parameters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

real_marker_size = 1.4  # Real size of the ArUco marker in cm

task_mode = 1
dimention_mode = "size"

color_ranges = {
    "white": ([0, 0, 230], [0, 0, 255]),    # Very strict white (high V, low S)
    "black": ([0, 0, 0], [0, 0, 75]),      # Very strict black (very low V, no S)
    "blue": ([94, 80, 2], [126, 255, 255]),    # Blue color range in HSV
    "green": ([30, 100, 100], [90, 255, 255]), # Green color range in HSV
    "red": ([0, 120, 70], [10, 255, 255]),     # Red (lower range for hue wrapping around)
    "yellow": ([15, 50, 20], [35, 255, 255]), # Yellow color range in HSV
}

# Open the camera feed
# cap = cv2.VideoCapture("https://193.10.201.62:8080")  # For IP camera
cap = cv2.VideoCapture(1)  # For webcam

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_mask = np.zeros(frame.shape[:2], dtype="uint8")

        # Convert frame to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        # Initialize scale factor
        scale_factor = None

        if ids is not None:
            # Draw detected markers
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate scale factor using the first detected ArUco marker
            pts = corners[0][0].astype(int)  # Use the first marker
            width = np.linalg.norm(pts[1] - pts[0])  # Top-right to Top-left
            scale_factor = real_marker_size / width  # Scale factor in cm/pixel

            # Draw green rectangle around each detected marker
            for corner in corners:
                pts = corner[0].astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # If scale factor is calculated, proceed with object detection
        if scale_factor is not None:
            # Perform edge detection or thresholding
            # edges = cv2.Canny(gray, 85, 190)  # Example with Canny edge detection
            edges = cv2.adaptiveThreshold(
                gray, 
                maxValue=255, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                thresholdType=cv2.THRESH_BINARY_INV, 
                blockSize=11, 
                C=4
            )
            cv2.imshow("Edges", edges)
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Ignore small contours
                if cv2.contourArea(contour) < 100:
                    continue

                # Create an empty image to draw contours on
                contour_image = np.zeros_like(frame)
                # Draw contours on contour_image
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
                # Display the contours in a new window
                cv2.imshow("Contours Window", contour_image)

                # Get the minimum area rectangle (rotated bounding box)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)  # Get the 4 corners of the rotated rectangle
                box = np.int32(box)  # Convert points to integer

                # Calculate the width and height (real-world size)
                width, height = rect[1]  # Size of the bounding rectangle

                real_width = width * scale_factor
                real_height = height * scale_factor

                min_dimention = min(real_width, real_height)
                max_dimention = max(real_width, real_height)

                if task_mode == 1 or task_mode == 2:
                    if ((max_dimention < 2.1 and max_dimention > 1.9) and (min_dimention < 0.5 and min_dimention > 0.3)
                            or (max_dimention < 2.5 and max_dimention > 2.3) and (min_dimention < 1.1 and min_dimention > 0.9)):
                        
                        if task_mode == 1:
                            for color, (lower, upper) in color_ranges.items():
                                lower_bound = np.array(lower, dtype="uint8")
                                upper_bound = np.array(upper, dtype="uint8")
                                mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
                                hsv_mask = cv2.bitwise_or(hsv_mask, mask)

                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
                                opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                                hsv_contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                # Display the hsv_mask in a separate window
                                cv2.imshow("HSV Frame", hsv_mask)

                                for hsv_contour in hsv_contours:
                                    if cv2.contourArea(hsv_contour) < 50:
                                        continue

                                    # Calculate the centroid of the contour
                                    M = cv2.moments(hsv_contour)
                                    if M["m00"] > 0:
                                        cX = int(M["m10"] / M["m00"])
                                        cY = int(M["m01"] / M["m00"])

                                        # Check if the centroid is inside the minimum area rectangle (rotated bounding box)
                                        if cv2.pointPolygonTest(box, (cX, cY), False) >= 0:
                                            # Put the color name on the original frame
                                            cv2.putText(frame, color, (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)    
                            
                            # Draw the rotated rectangle on the original frame
                            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
                            
                            if dimention_mode == "size":
                                size_text = "Large" if min_dimention > 0.9 else "Small"
                                leftmost_point = sorted(box, key=lambda x: x[0])[0]
                                cv2.putText(frame, size_text, (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            if dimention_mode == "dimention":
                                # Display the dimensions on the frame
                                dimension_text = f"{real_width:.2f}x{real_height:.2f} cm"
                                leftmost_point = sorted(box, key=lambda x: x[0])[0]
                                cv2.putText(frame, dimension_text, (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if task_mode == 2:
                            # Get the point with the minimum y value
                            min_y_point = min(box, key=lambda p: p[0])
                            # Extract the x-coordinate of the point with the minimum y value
                            min_y_x = min_y_point[1]

                            # Count the points with x-coordinates smaller than min_y_x
                            count_x_smaller = sum(1 for p in box if p[1] < min_y_x)
                            
                            ellipse = cv2.fitEllipse(contour)

                            # Draw the ellipse on the image
                            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                            
                            if count_x_smaller == 2:
                                orientation = 90 - ellipse[2]
                                orientation_text = str(round(orientation)) + " degrees"
                                leftmost_point = sorted(box, key=lambda x: x[0])[0]
                                cv2.putText(frame, orientation_text, (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1)
                            elif count_x_smaller == 1:
                                orientation = 270 - ellipse[2]
                                orientation_text = str(round(orientation)) + " degrees"
                                leftmost_point = sorted(box, key=lambda x: x[0])[0]
                                cv2.putText(frame, orientation_text, (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1)

                if task_mode == 3:
                    if (max_dimention < 2.2) and (min_dimention < 1.1):
                        aspect_ratio = min_dimention / max_dimention
                        if ((aspect_ratio < 0.36 or aspect_ratio > 0.48) and min_dimention > 1) or ((aspect_ratio < 0.18 or aspect_ratio > 0.22) and max_dimention < 1.7):
                            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
                            leftmost_point = sorted(box, key=lambda x: x[0])[0]
                            cv2.putText(frame, "defected", (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
        # Display the original frame with bounding rectangles and dimensions
        cv2.imshow('Object Dimension Detection', frame)

        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            task_mode = 1
        elif key == ord('2'):
            task_mode = 2
        elif key == ord('3'):
            task_mode = 3
        elif key == ord('s'):
            dimention_mode = "size"
        elif key == ord('d'):
            dimention_mode = "dimention"

finally:
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
