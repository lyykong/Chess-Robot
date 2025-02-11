import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to select points manually on the image
def select_points(image):
    points = []

    # Function to capture points when the mouse is clicked
    def click_event(event, x, y, flags, params):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot on the selected point
            cv2.imshow("Select Points", image)
            if len(points) == 4:  # Stop after selecting 4 points
                print("4 points selected.")
                cv2.destroyAllWindows()

    # Show the image and wait for the user to click 4 points
    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", click_event)
    cv2.waitKey(0)

    return points

# Function to compute the perspective transformation matrix
def compute_perspective_transform(image, points):
    # Define the destination points (square corners)
    dst_points = np.array([
        [0, 0],        # top-left
        [image.shape[1] - 1, 0],  # top-right
        [image.shape[1] - 1, image.shape[0] - 1],  # bottom-right
        [0, image.shape[0] - 1]  # bottom-left
    ], dtype="float32")

    # Convert the points to a NumPy array (image coordinates)
    src_points = np.array(points, dtype="float32")

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

# Function to detect specific colors close to green (#4A8227) or red (#88450E)
def detect_specific_colors_in_center(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the target green (#4A8227) in RGB -> HSV
    target_green_rgb = np.array([74, 130, 39])  # #4A8227 RGB
    target_green_hsv = cv2.cvtColor(np.uint8([[target_green_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    # Define the target red (#88450E) in RGB -> HSV
    target_red_rgb = np.array([136, 69, 14])  # #88450E RGB
    target_red_hsv = cv2.cvtColor(np.uint8([[target_red_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    # Define hue tolerance range
    hue_tolerance = 10  # A range of +/- 10 degrees in hue is allowed

    # Define the green range in HSV
    lower_green = np.array([target_green_hsv[0] - hue_tolerance, 100, 100])
    upper_green = np.array([target_green_hsv[0] + hue_tolerance, 255, 255])

    # Define the red range in HSV
    lower_red = np.array([target_red_hsv[0] - hue_tolerance, 100, 100])
    upper_red = np.array([target_red_hsv[0] + hue_tolerance, 255, 255])

    # Create masks for green and red colors
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Get the center region (50% of the image)
    height, width, _ = image.shape
    center_height_start = height // 4
    center_height_end = 3 * height // 4
    center_width_start = width // 4
    center_width_end = 3 * width // 4
    
    # Crop the image to focus only on the center region
    center_green_mask = green_mask[center_height_start:center_height_end, center_width_start:center_width_end]
    center_red_mask = red_mask[center_height_start:center_height_end, center_width_start:center_width_end]

    # Calculate the number of green and red pixels in the center region
    center_green_pixels = np.sum(center_green_mask)
    center_red_pixels = np.sum(center_red_mask)
    center_total_pixels = center_green_mask.size  # Total pixels in the center region

    # Calculate the proportion of green and red pixels in the center region
    center_green_ratio = center_green_pixels / center_total_pixels
    center_red_ratio = center_red_pixels / center_total_pixels

    # Determine if the color detection threshold is met in the center region
    if center_red_ratio >= 0.15:
        return 5  # Color close to red detected in center (>= 15% of center pixels)
    elif center_green_ratio >= 0.15:
        return 6  # Color close to green detected in center (>= 15% of center pixels)
    else:
        return 0  # No red or green detected in the center region (less than 15%)

# Function to process the cropped image and split it into sub-images
def process_cropped_image(cropped_image):
    height, width, _ = cropped_image.shape
    rows = 8
    cols = 8
    sub_height = height // rows
    sub_width = width // cols
    sub_images = []

    for i in range(rows):
        for j in range(cols):
            y_start = i * sub_height
            y_end = (i + 1) * sub_height
            x_start = j * sub_width
            x_end = (j + 1) * sub_width
            sub_image = cropped_image[y_start:y_end, x_start:x_end]
            sub_images.append(sub_image)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flatten()):
        label = detect_specific_colors_in_center(sub_images[i])
        ax.imshow(sub_images[i])
        ax.axis('off')
        color_map = {5: 'Red Color', 6: 'Green Color'}
        ax.set_title(f"Sub-image {i+1} - {color_map.get(label, 'None')}")

    plt.tight_layout()
    plt.show()

# Function to crop the image based on the selected four points (before warp)
def crop_around_corners(image, corners):
    x_coords = [corner[0] for corner in corners]
    y_coords = [corner[1] for corner in corners]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image

# Load the original image
image_path = r'C:\Users\User\Documents\Engineering Group Project\Program Chess\thumbnail_test_chess2.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: The image at path {image_path} could not be loaded.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Manually select the 4 corner points from the image
selected_points = select_points(image_rgb)

# Ensure 4 points are selected
if len(selected_points) != 4:
    print("Error: You must select exactly 4 points.")
    exit()

# Compute the perspective transformation matrix
matrix = compute_perspective_transform(image_rgb, selected_points)

# Crop the image around the selected corners (before warp)
cropped_image = crop_around_corners(image_rgb, selected_points)

# Process the cropped image (split it and detect red and green colors)
process_cropped_image(cropped_image)
