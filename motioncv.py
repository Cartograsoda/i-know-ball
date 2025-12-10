import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob


OUTPUT_DIRS = {
    "RAW": "1_Input",
    "BG_SUBTRACTED": "2_BackgroundRemoved",
    "CLEANED": "3_Cleaned",
    "DETECTED": "4_Detected"
}


def setup_environment():
    """
    Creates output directories if they don't exist.
    """
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    print("Output directories created/verified")





def get_image_list(data_folder):
    """
    Gets a sorted list of image files from the data folder.
    """
    # Support common image formats
    image_extensions = ['*.jpg']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_folder, extension)))
        image_files.extend(glob.glob(os.path.join(data_folder, extension.upper())))
    
    return sorted(image_files)

def get_initial_background(image_files):
    """

    1. Read the very first 5 image from the list.
    2. Convert this frame to Grayscale (cv2.cvtColor).
    3. Apply Gaussian Blur to reduce noise
    4. Store this as the reference

    """
    if not image_files:
        return None
    
    # Read first 5 images and calculate average
    background_frames = []
    
    for i in range(5):
        frame = cv2.imread(image_files[i])
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        background_frames.append(blurred_frame)
    
    # Calculate average of all 5 background frames
    background_avg = np.mean(background_frames, axis=0).astype(np.uint8)
    return background_avg

def preprocess_frame(frame):
    """
    Prepares a single video frame for comparison.
    1. Convert the incoming frame to Grayscale.
    2. Apply Gaussian Blur (same parameters as background) to match the reference.

    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    return blurred_frame

def remove_background_and_threshold(current_gray, reference_gray):
    """
        
    Implementation Logic:
    1. Background Removal: Calculate absolute difference using cv2.absdiff(reference, current).
    2. Thresholding: Apply cv2.threshold to the difference image.
       thresh=25, maxval=255 to create a binary mask.
    
    
    """
    diff = cv2.absdiff(reference_gray, current_gray)
    _, binary_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return binary_mask

def denoise_mask(binary_mask):
    """

    Implementation Logic:
    1. Define a kernel (e.g., 5x5 matrix).
    2. Aim: Remove small white noise dots
    
    """
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.erode(binary_mask, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    return cleaned

def calculate_center(cleaned_mask):
    """
    Finds the center of the moving object (ball).

        
    Implementation Logic:
    1. Find contours using cv2.findContours.
    2. Identify the largest contour (assuming the ball is the largest moving object).
    3. Derive center (cx, cy) using m00 m01 m10
    
    """
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    
    if M['m00'] == 0:
        return None
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    return (cx, cy)

def save_step_images(frame_id, input_img, bg_removed_img, cleaned_img, detected_img):
    """
    save images after processing
    """
    filename = "frame_" + str(frame_id).zfill(3) + ".jpg"
    
    cv2.imwrite(os.path.join(OUTPUT_DIRS["RAW"], filename), input_img)
    cv2.imwrite(os.path.join(OUTPUT_DIRS["BG_SUBTRACTED"], filename), bg_removed_img)
    cv2.imwrite(os.path.join(OUTPUT_DIRS["CLEANED"], filename), cleaned_img)
    cv2.imwrite(os.path.join(OUTPUT_DIRS["DETECTED"], filename), detected_img)

def plot_tracking_data(x_history, y_history):
   
    plt.figure(figsize=(12, 9))
    
    # Plot the trajectory
    plt.plot(x_history, y_history, 'b-', marker='o', markersize=4, linewidth=2, label='Ball trajectory')
    
    # Set axis limits to match image dimensions
    plt.xlim(0, 320)  # Image width
    plt.ylim(0, 240)  # Image height
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    
    # Add coordinate annotations for each point
    for i, (x, y) in enumerate(zip(x_history, y_history)):
        plt.annotate("(" + str(x) + "," + str(y) + ")", (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.title('Ball Center Tracking (Image Coordinates: 320x240)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add frame numbers as secondary info
    if len(x_history) > 1:
        plt.text(0.02, 0.98, "Frames tracked: " + str(len(x_history)), 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():

    setup_environment()
    
    # Get list of images from data folder
    data_folder = "data"
    image_files = get_image_list(data_folder)
    
    if not image_files:
        print("Error: No images found in " + data_folder + " folder")
        return
    
    print("Found " + str(len(image_files)) + " images")
    
    reference_bg = get_initial_background(image_files)
    if reference_bg is None:
        print("Error: Could not get background frame")
        return
    
    ball_x = []
    ball_y = []
    
    for frame_id, image_path in enumerate(image_files, 1):
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print("Warning: Could not load image " + image_path)
            continue
        
        print("Processing frame " + str(frame_id) + "/" + str(len(image_files)))
        
        # Process frame
        processed_frame = preprocess_frame(frame)
        raw_mask = remove_background_and_threshold(processed_frame, reference_bg)
        clean_mask = denoise_mask(raw_mask)
        
        # Calculate center
        center = calculate_center(clean_mask)
        
        # Create detected frame for visualization (side-by-side comparison)
        detected_with_circle = frame.copy()
        
        if center is not None:
            cx, cy = center
            ball_x.append(cx)
            ball_y.append(cy)
            cv2.circle(detected_with_circle, (cx, cy), 10, (255, 0, 0), 2)
            print("  Ball detected at (" + str(cx) + ", " + str(cy) + ")")
        else:
            print("  No ball detected in frame " + str(frame_id))
        
        # Create side-by-side comparison image
        height, width = frame.shape[:2]
        combined_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Place original on left side
        combined_frame[:, :width] = frame
        
        # Place detected version on right side  
        combined_frame[:, width:] = detected_with_circle
        
        # Add labels
        cv2.putText(combined_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, "Detected", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save step images
        save_step_images(frame_id, frame, raw_mask, clean_mask, combined_frame)
    
    # Plot tracking data
    if ball_x and ball_y:
        plot_tracking_data(ball_x, ball_y)
        print("Tracked ball in " + str(len(ball_x)) + " frames")
        print("X coordinates: " + str(ball_x))
        print("Y coordinates: " + str(ball_y))
    else:
        print("No ball detected in any frame")

if __name__ == "__main__":
    main()