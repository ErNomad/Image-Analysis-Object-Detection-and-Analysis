import cv2
import cv2.aruco as aruco
import numpy as np

# Define the ArUco dictionary and parameters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

# Real size of the ArUco marker in cm.
real_marker_size = 1.4 

'''
Define modes to be able to change the frame being displayed based on the task.
Mode 1 is for needle detection task. Mode 2 for showing the orientation of
the needles and mode 3 is for defective needle detection.
Keyboard buttons 1, 2 and 3 can be used to switch between modes 1, 2 and 3 respectively.
The default is needle detection mode.

The keyboard buttons 's' and 'd' can be used to display the needle size as large or small,
or to show the needle dimensions.
The default is to show the size as being large or small.
'''
task_mode = 1
dimension_mode = "size"

# Define a dictionary for the needle color ranges in HSV.
color_ranges = {
    "white": ([0, 0, 230], [0, 0, 255]),
    "black": ([0, 0, 0], [0, 0, 75]),
    "blue": ([94, 80, 2], [126, 255, 255]),
    "green": ([30, 100, 100], [90, 255, 255]),
    "red": ([0, 120, 70], [10, 255, 255]),
    "yellow": ([15, 50, 20], [35, 255, 255]),
}

# Open the camera feed. "1" as the argument is for using the portable USB camera.
cap = cv2.VideoCapture(1)

# Check if a camera is present or usable.
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

try:
    while True:
        # Break out of the loop if capturing the video stream is not successful.
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_mask = np.zeros(frame.shape[:2], dtype="uint8")

        # Convert frame to grayscale for ArUco detection.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame.
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

        # Initialize scale factor.
        scale_factor = None

        if ids is not None:
            # Draw detected markers.
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate scale factor using the first detected ArUco marker.
            pts = corners[0][0].astype(int)  # Use the first marker.
            width = np.linalg.norm(pts[1] - pts[0])  # Top-right to Top-left.
            scale_factor = real_marker_size / width  # Scale factor in cm/pixel.

            # Draw green rectangle around each detected marker.
            for corner in corners:
                pts = corner[0].astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # If scale factor is calculated, proceed with object detection.
        if scale_factor is not None:
            # Perform adaptive thresholding
            edges = cv2.adaptiveThreshold(
                gray, 
                maxValue=255, 
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                thresholdType=cv2.THRESH_BINARY_INV, 
                blockSize=11, 
                C=4
            )
            # Show the result of adaptive thresholding in a separate window.
            cv2.imshow("Edges", edges)
            # Find contours.
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Ignore small contours.
                if cv2.contourArea(contour) < 100:
                    continue

                # Create an empty image to draw contours on.
                contour_image = np.zeros_like(frame)
                # Draw contours on contour_image.
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
                # Display the contours in a new window.
                cv2.imshow("Contours Window", contour_image)

                '''
                Get the minimum area rectangle (rotated bounding box). A minimum area rectangle
                is used instead of a bounding box so that objects can be detected based on the
                rectangle's dimensions regardless of their orientation.
                '''
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)  # Get the 4 corners of the rotated rectangle.
                box = np.int32(box)  # Convert points to integer.

                # Calculate the width and height (in pixels).
                width, height = rect[1]

                # Calculate real world dimentions in cm.
                real_width = width * scale_factor
                real_height = height * scale_factor
                
                '''
                To be able to detect the objects based on their size, minimum and Maximum
                dimentions among the real dimentions are found. The reason is that when
                using the 'minAreaRect' function, depending on the orientation of the object,
                OpenCV may assign the larger side to either of the 'width' or 'height'
                variables. This will be problematic when comparing the dimention range of the
                objects to be detected (needles) with width and height values since we won't be
                able to know which variable should be compared to the large side of the needles
                and which one should be compared to the small side of the needles as the orientation
                changes. By finding the 'min_dimention' and 'max_dimention' variables for each min
                area rectangle, we can solve this problem by comparing the large dimantion range
                and small dimention range with 'max_dimention' and 'min_dimention' variables respectively.
                '''
                min_dimention = min(real_width, real_height)
                max_dimention = max(real_width, real_height)

                # If the objective is to detect the objects or to find the orientations.
                if task_mode == 1 or task_mode == 2:
                    '''
                    Determine if the rotated bounding box dimensions match the size the needles. If the
                    dimensions are not within these ranges, the objects will be ignored. The tolerances
                    used are within the ±1mm precision range indicated in the assignment instructions.
                    '''
                    if ((max_dimention < 2.1 and max_dimention > 1.9) and (min_dimention < 0.5 and min_dimention > 0.3)
                            or (max_dimention < 2.5 and max_dimention > 2.3) and (min_dimention < 1.1 and min_dimention > 0.9)):
                        # Needle detection mode.
                        if task_mode == 1:
                            '''
                            Iterate through the 'color_ranges' dictionary to create a mask for each
                            color based on its HSV range. This process is similar to thresholding,
                            where the resulting mask is a binary image that highlights the pixels
                            corresponding to that specific color as foreground pixels.

                            The masks for all colors are then combined using a bitwise OR operation.
                            '''
                            for color, (lower, upper) in color_ranges.items():
                                lower_bound = np.array(lower, dtype="uint8")
                                upper_bound = np.array(upper, dtype="uint8")
                                mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
                                hsv_mask = cv2.bitwise_or(hsv_mask, mask)
                                
                                # Performing an opening operation to remove the long, thin parts of
                                # the needles, as they are often detected as white areas due to
                                # light reflection.
                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
                                opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                                # Find the contours of the color masks.
                                hsv_contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                # Display the hsv_mask in a separate window.
                                cv2.imshow("HSV Frame", hsv_mask)

                                for hsv_contour in hsv_contours:
                                    # Ignore small contours.
                                    if cv2.contourArea(hsv_contour) < 50:
                                        continue
                                    
                                    '''
                                    To display only the color of the detected needles and not
                                    other objects, the centroid of the larger parts of the needles
                                    is calculated. If the centroid lies within the boundaries of
                                    the minimum area rectangle surrounding the needle, it is
                                    assumed that the color is associated with a needle and is
                                    displayed on the frame.
                                    '''
                                    # Calculate the centroid of the contour.
                                    M = cv2.moments(hsv_contour)
                                    if M["m00"] > 0:
                                        cX = int(M["m10"] / M["m00"])
                                        cY = int(M["m01"] / M["m00"])

                                        # Check if the centroid is inside the minimum area rectangle (rotated bounding box).
                                        if cv2.pointPolygonTest(box, (cX, cY), False) >= 0:
                                            # Put the color name on the original frame.
                                            cv2.putText(frame, color, (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)    
                            
                            # Draw the rotated rectangle on the original frame.
                            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
                            
                            # Display either the size or the dimensions.
                            if dimension_mode == "size":
                                # If the small side of the rotated bounding box of the detected needle
                                # is larger than 0.9 cm, it is a large needle. Otherwise it is a small one.
                                size_text = "Large" if min_dimention > 0.9 else "Small"
                                leftmost_point = sorted(box, key=lambda x: x[0])[0]
                                cv2.putText(frame, size_text, (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            if dimension_mode == "dimension":
                                dimension_text = f"{real_width:.2f}x{real_height:.2f} cm"
                                leftmost_point = sorted(box, key=lambda x: x[0])[0]
                                cv2.putText(frame, dimension_text, (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # Orientation calculation mode.
                        if task_mode == 2:
                            '''
                            The bounding elipse of the objects is used to calculate the orientation
                            of detected needles. The third element in the array returned by fitEllipse
                            function is an orientation value. But since we want to calculate the
                            orientation according to the positive horizontal direction of the Cartesian
                            axis system, the following calcualtions are done.
                            Orientation value obtained from fitEllipse function is dependant on the
                            orientation of the object. If the object is in a position which the angle
                            is obtuse, then the function returnes a value relative to the negative vertical
                            direction of the Cartesian axis system and if it is acute the angle is relative to
                            the positive vertical direction, both clockwise. So there is a need to know if the
                            orientation of the object is acute or obtuse. For this, we can use the min area
                            rectangle. If the leftmost point of this rectangle is lower than two points in x direction, then
                            the angle is acute, otherwise, if it is lower than one point on the rectangle,
                            then the angle is obtuse. Then we can decide whether we need to subract the angle
                            returned from the function from 270 or 90 degrees.
                            '''
                            # Get the point with the minimum y value.
                            min_y_point = min(box, key=lambda p: p[0])
                            # Extract the x-coordinate of the point with the minimum y value.
                            min_y_x = min_y_point[1]

                            # Count the points with x-coordinates smaller than min_y_x.
                            count_x_smaller = sum(1 for p in box if p[1] < min_y_x)
                            
                            # Calculate the ellipse for detected needles.
                            ellipse = cv2.fitEllipse(contour)

                            # Draw the ellipse on the image.
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
                                
                # Defective needle detection mode.
                if task_mode == 3:
                    # Considering a region of interest based on bounding boxes equal or smaller than the size of a large needle.
                    if (max_dimention < 2.2) and (min_dimention < 1.1):
                        # Calculate the aspect ration of the needles inside the region of interest.
                        aspect_ratio = min_dimention / max_dimention
                        '''
                        Determin if the aspect ratio is not in the range of aspect ratios acceptable
                        for a non-defective needle. If the condition is satisfied, the needle is
                        considered to be defective. The aspect ratio conditions are combined with
                        minimum and maximum dimensions of the rotated bounding boxes so that the 
                        aspect ratios for needles with different sizes do not overlap when deciding
                        whether a needle is defective or not.
                        '''
                        if ((aspect_ratio < 0.36 or aspect_ratio > 0.48) and min_dimention > 1) or ((aspect_ratio < 0.18 or aspect_ratio > 0.22) and max_dimention < 1.7):
                            cv2.polylines(frame, [box], isClosed=True, color=(255, 0, 0), thickness=2)
                            leftmost_point = sorted(box, key=lambda x: x[0])[0]
                            cv2.putText(frame, "defective", (leftmost_point[0] - 20, leftmost_point[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
        # Display the frame.
        cv2.imshow('Object Dimension Detection', frame)

        # Exit the loop when 'q' is pressed or switch between different modes.
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
            dimension_mode = "size"
        elif key == ord('d'):
            dimension_mode = "dimension"

finally:
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
