
# %%
import cv2
import numpy as np
import os

### Step 1: Initialization & Setup

input_video = '3.mp4'

# Open the video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise Exception(f"Error: Could not open {input_video}")

# Get video properties for saving outputs
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0 # Fallback FPS

# Define codec and create VideoWriter objects for the 4 stages
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_morph  = cv2.VideoWriter('3_2_morph_cleaned.mp4', fourcc, fps, (width, height), isColor=False)
out_diff   = cv2.VideoWriter('3_0_color_diff.mp4', fourcc, fps, (width, height), isColor=True)
out_skel   = cv2.VideoWriter('3_3_medial_axis.mp4', fourcc, fps, (width, height), isColor=False)
out_hough  = cv2.VideoWriter('3_4_hough_axis.mp4', fourcc, fps, (width, height), isColor=True)

print("Calculating median background from a sample of frames...")
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Sample 30 evenly spaced frames across the video to build the median model
if frame_count_total > 0:
    frame_indices = np.linspace(0, frame_count_total - 1, 30, dtype=int)
else:
    # Fallback if frame count is unavailable
    frame_indices = range(30)
    
frames = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        break # End of video reached
    
    # Keep the frame in full BGR color for color median background
    frames.append(frame)
        
if not frames:
    raise Exception("Could not read any frames to calculate background.")

# Compute the median along the time axis (now across 3 color channels)
median_bg = np.median(frames, axis=0).astype(np.uint8)


# Reset video back to the first frame for the main processing loop
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("Median background calculated successfully.")
cv2.imwrite("Median Background.jpg", median_bg)
# %% [markdown]
# ### Step 2: Helper Function for Skeletonization
# OpenCV doesn't have a direct "medial axis" function built into its core module, 
# so we use morphological thinning (iterative erosion) to find the skeleton.

# %%
def get_medial_axis(binary_mask):
    """
    Reduces a binary mask to a 1-pixel wide skeleton (medial axis)
    using iterative morphological operations.
    """
    skeleton = np.zeros(binary_mask.shape, np.uint8)
    img = binary_mask.copy()
    size = np.size(img)
    
    # Cross-shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    done = False
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        # Subtracting the open image from the original image
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        
        # SAFEGUARD: Break if erosion stops changing the image to prevent infinite loops
        if np.array_equal(img, eroded):
            break
            
        img = eroded.copy()
        
        # Stop when image is completely eroded
        if cv2.countNonZero(img) == 0:
            break
            
    return skeleton

# ### Step 3: Main Processing Loop
# Read frames, process them through the 4 stages, and write to output videos.

def custom_hough_lines(image, rho_res, theta_res, threshold):
    """
    Custom vectorized implementation of Standard Hough Line Transform with Non-Maximum Suppression (NMS).
    Returns lines in the same nested array format as cv2.HoughLines: [[[rho, theta]]]
    """
    height, width = image.shape
    # Maximum possible distance from origin is the image diagonal
    diag_len = int(np.ceil(np.sqrt(width**2 + height**2)))
    
    # Setup arrays for thetas and rhos
    thetas = np.arange(0, np.pi, theta_res)
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    
    # Initialize the Hough accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # Find all white pixels (y, x coordinates)
    y_idxs, x_idxs = np.nonzero(image)
    if len(x_idxs) == 0:
        return None
        
    # Vectorized computation of rho = x*cos(theta) + y*sin(theta)
    x_col = x_idxs[:, np.newaxis]
    y_col = y_idxs[:, np.newaxis]
    cos_row = np.cos(thetas)[np.newaxis, :]
    sin_row = np.sin(thetas)[np.newaxis, :]
    
    rho_vals = x_col * cos_row + y_col * sin_row
    rho_idxs = np.round((rho_vals + diag_len) / rho_res).astype(int)
    
    flat_rho_idxs = rho_idxs.flatten()
    flat_theta_idxs = np.tile(np.arange(len(thetas)), len(x_idxs))
    
    # Vote in the accumulator
    np.add.at(accumulator, (flat_rho_idxs, flat_theta_idxs), 1)
    
    # Extract peaks that exceed the vote threshold
    rho_peaks, theta_peaks = np.where(accumulator >= threshold)
    if len(rho_peaks) == 0:
        return None
        
    # Get the vote counts to sort the lines (most dominant first)
    votes = accumulator[rho_peaks, theta_peaks]
    sort_idxs = np.argsort(votes)[::-1]
    
    # --- NON-MAXIMUM SUPPRESSION (NMS) ---
    lines = []
    for idx in sort_idxs:
        r_idx = rho_peaks[idx]
        t_idx = theta_peaks[idx]
        r_val = rhos[r_idx]
        t_val = thetas[t_idx]
        
        
        lines.append([[r_val, t_val]])
            
    return np.array(lines, dtype=np.float32) if lines else None
# %%
print("Processing video...")
bg_diff_threshold = 30 

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video
    
    frame_count += 1
    
    # ---------------------------------------------------------
    # STAGE 1: Color Median Background Subtraction
    # ---------------------------------------------------------
    
    # Compute the absolute difference between the current color frame and color median background
    # This results in a 3-channel difference image (B_diff, G_diff, R_diff)
    diff_color = cv2.absdiff(median_bg, frame)
    out_diff.write(diff_color)  # Save the color difference for debugging
    # Convert the 3-channel difference image to grayscale
    # This effectively combines the differences from all channels into a single intensity map
    diff_gray = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
    

    # Threshold the combined difference to get a pure binary mask
    _, binary_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    out_fg.write(binary_mask)

    
    # ---------------------------------------------------------
    # STAGE 2: Morphological Cleaning
    # ---------------------------------------------------------
    # Remove noise (Opening) and fill holes (Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 1. Opening: removes small noise points in the background
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # 2. Closing: fills small holes inside the foreground object
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    # 3. Dilation (Optional but recommended): smooths edges slightly before skeletonization   
    cleaned = cv2.dilate(cleaned, kernel, iterations=1) 
    out_morph.write(cleaned)
    
    # ---------------------------------------------------------
    # STAGE 3: Medial Axis (Skeletonization)
    # ---------------------------------------------------------
    skeleton = get_medial_axis(binary_mask)
    out_skel.write(skeleton)
    
    # ---------------------------------------------------------
    # STAGE 4: Hough Line Transform on the Medial Axis
    # ---------------------------------------------------------
    # Overlay lines on a copy of the original frame
    output_frame = frame.copy()
    
    # Parameters: (image, rho, theta, threshold, minLineLength, maxLineGap)
    # You will likely need to tune threshold, minLineLength, and maxLineGap 
    # based on the size of your object in the frame.
    # Get all (x, y) coordinates of the white skeleton pixels
    lines = custom_hough_lines(skeleton, 1, np.pi / 180, 50)
    
    min_separation = 20
    
    if lines is not None and len(lines) > 0:
        use_two_lines = False
        rho2, theta2 = None, None
        # The strongest line is our primary anchor
        rho1, theta1 = lines[0][0]
        if len(lines) >= 2:
            # Iterate through the remaining lines to find a valid second edge
            for i in range(1, len(lines)):
                rho_candidate, theta_candidate = lines[i][0]
                
                # FIX WRAP-AROUND between the anchor and candidate for distance calculation
                diff_t = theta_candidate - theta1
                if diff_t > np.pi / 2:
                    theta_candidate -= np.pi
                    rho_candidate = -rho_candidate
                elif diff_t < -np.pi / 2:
                    theta_candidate += np.pi
                    rho_candidate = -rho_candidate
                    
                # NEW PHYSICAL DISTANCE CHECK: Measure X-distance at the middle of the screen
                y_mid = height / 4.0
                cos1 = np.cos(theta1) if abs(np.cos(theta1)) > 1e-5 else 1e-5
                cos_cand = np.cos(theta_candidate) if abs(np.cos(theta_candidate)) > 1e-5 else 1e-5
                
                x1_mid = (rho1 - y_mid * np.sin(theta1)) / cos1
                x_cand_mid = (rho_candidate - y_mid * np.sin(theta_candidate)) / cos_cand
                
                # Verify lines are roughly parallel (e.g., within 5 degrees)
                is_parallel = abs(theta1 - theta_candidate) < (7 * np.pi / 180)
                
                # Check if this candidate is physically separated AND roughly parallel
                if  60 >= abs(x1_mid - x_cand_mid) >= min_separation and is_parallel:
                    rho2 = rho_candidate
                    theta2 = theta_candidate
                    use_two_lines = True
                    break # Found the valid opposite edge! Stop iterating.
                
        if use_two_lines:
            # Average them to get the medial axis
            rho_raw = rho1 - (x1_mid - x_cand_mid)/2.0 * np.cos(theta1) # Adjust rho based on the actual physical midpoint between the two lines
            theta_raw = theta1 

            # Identify the "Right" edge (the one with the larger X coordinate at the middle of the screen)
            y_mid = height / 2.0
            cos1 = np.cos(theta1) if abs(np.cos(theta1)) > 1e-5 else 1e-5
            cos2 = np.cos(theta2) if abs(np.cos(theta2)) > 1e-5 else 1e-5
            
            x1_mid = (rho1 - y_mid * np.sin(theta1)) / cos1
            x2_mid = (rho2 - y_mid * np.sin(theta2)) / cos2
            
        else:
            # FALLBACK: Only one line found (or no valid left edge was far enough away)! 
            rho_raw, theta_raw = lines[0][0]
            
            
        
        
        a = np.cos(theta1)
        b = np.sin(theta1)
        x0 = a * rho1
        y0 = b * rho1
        
        mult = max(width, height) * 2
        
        pt1 = (int(x0 + mult * (-b)), int(y0 + mult * (a)))
        pt2 = (int(x0 - mult * (-b)), int(y0 - mult * (a)))
        #cv2.line(output_frame, pt1, pt2, (255, 0, 0), 3)
        # Convert polar to Cartesian to draw the medial axis
        a = np.cos(theta_raw)
        b = np.sin(theta_raw)
        x0 = a * rho_raw
        y0 = b * rho_raw
        
        mult = max(width, height) * 2
        
        pt1 = (int(x0 + mult * (-b)), int(y0 + mult * (a)))
        pt2 = (int(x0 - mult * (-b)), int(y0 - mult * (a)))
        
        cv2.line(output_frame, pt1, pt2, (0, 0, 255), 5) # Red line for medial axis
        if(theta2 is not None):
            a = np.cos(theta2)
            b = np.sin(theta2)
            x0 = a * rho2
            y0 = b * rho2
            
            mult = max(width, height) * 2
            
            pt1 = (int(x0 + mult * (-b)), int(y0 + mult * (a)))
            pt2 = (int(x0 - mult * (-b)), int(y0 - mult * (a)))
            #cv2.line(output_frame, pt1, pt2, (0, 255, 0), 3)
                        
            
    out_hough.write(output_frame)

print(f"Finished processing {frame_count} frames.")

# %% [markdown]
# ### Step 4: Cleanup
# Release the capture and writer objects to save the files properly.

# %%
cap.release()
out_fg.release()
out_morph.release()
out_skel.release()
out_hough.release()
print("All video stages saved successfully.")

# %%