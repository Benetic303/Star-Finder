import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from astropy.stats import mad_std
from scipy.ndimage import maximum_filter # Useful for finding local maxima
from scipy.ndimage import label, find_objects
import sys


def load_grayscale_image(image_path):
    try:

        # load image in grayscale
        image_data = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if image_data is None:
            raise FileNotFoundError(f"Could not open or find the image at {image_path}")
        return image_data

    except Exception as e:
        print(f"Error loading PNG with OpenCV {image_path}: {e}")
        return None


def estimate_background_and_noise(image_array):
    # A simple approach: median and std of all pixels
    median_val = np.median(image_array)
    # Using Median Absolute Deviation (MAD) is more robust for astronomical images
    # because it's less affected by bright stars (outliers) than standard deviation.
    noise_std = mad_std(image_array)

    return median_val, noise_std


def calculate_centroids_and_fluxes(image_array, threshold_value):
    #binary_image consist of all potential regions which could be star
    binary_image = image_array > threshold_value


    # 'label' finds groups of connected True pixels and assigns a unique integer ID to each group.
    # labeled_array: An array with the same shape as binary_image, where each pixel
    #                has an integer label (1, 2, 3...) corresponding to its component.
    # num_features: The total count of unique components found.
    labeled_array, num_features = label(binary_image)
    print(f"Found {num_features} connected components (potential stars).")

    all_obj_slices = find_objects(labeled_array)

    star_results = []  # List to store dictionaries of star information

    for i in range(1, num_features + 1):
        obj_slice = all_obj_slices[i - 1]

        # Ensure the slice is valid (can be None if find_objects was called with max_label)
        if obj_slice is None:
            continue


        # Extract the region of interest (ROI) for this object from the original image
        # This gets the actual pixel values for the current star's blob.
        star_roi = image_array[obj_slice]
        # Create a mask for the current object *within* its ROI.
        # This tells us which pixels in star_roi belong to the current star (label 'i').
        labeled_roi_mask = labeled_array[obj_slice] == i

        # Ensure the ROI is not empty and has relevant pixels
        if star_roi.size == 0 or np.sum(labeled_roi_mask) == 0:
            continue

        # Calculate Total Brightness (Flux)
        # Sum all pixel values within the masked region of the ROI.
        total_flux = np.sum(star_roi[labeled_roi_mask])

        # Avoid division by zero if a blob somehow has zero flux (e.g., all pixels are 0)
        if total_flux == 0:
            continue

        # Calculate Centroid (weighted average of pixel positions)
        # Get relative coordinates (y, x) of pixels within the ROI that belong to the current star.
        y_coords_in_roi, x_coords_in_roi = np.where(labeled_roi_mask)

        # Calculate weighted centroid relative to the ROI's top-left corner:
        # Sum (coordinate * pixel_value) for all pixels in the blob, then divide by total flux.
        weighted_x_in_roi = np.sum(x_coords_in_roi * star_roi[labeled_roi_mask]) / total_flux
        weighted_y_in_roi = np.sum(y_coords_in_roi * star_roi[labeled_roi_mask]) / total_flux

        # Convert relative ROI centroid to global image coordinates:
        # Add the starting row/column of the ROI back to the relative centroid.
        global_x = weighted_x_in_roi + obj_slice[1].start  # obj_slice[1] is the X slice
        global_y = weighted_y_in_roi + obj_slice[0].start  # obj_slice[0] is the Y slice

        star_results.append({
            'x_pixel': global_x,
            'y_pixel': global_y,
            'flux': total_flux,
            'peak_value': np.max(star_roi[labeled_roi_mask]),  # Brightest pixel in this specific blob
            'area_pixels': np.sum(labeled_roi_mask)  # Number of pixels in this specific blob
        })

    # Sort stars by brightness (flux) in descending order (brightest first)


    star_results.sort(key=lambda s: s['flux'], reverse=True)
    return star_results


def main():
    video = cv.VideoCapture("data/205427-926957416_small.mp4")
    if (video.isOpened() == False):
        print("Error opening video stream or file")
        sys.exit(1)  # Exit if video cannot be opened
    cv.namedWindow('StarFinder', cv.WINDOW_NORMAL)

    # Read the entire file until it is completed
    while (True):
        # Capture each frame
        ret, frame = video.read()
        #image_data = load_grayscale_image(frame)
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame_float = gray_frame.astype(np.float32)

        background, noise = estimate_background_and_noise(gray_frame_float)
        print(f"Estimated Background Median: {background:.2f}")
        print(f"Estimated Noise Standard Deviation: {noise:.2f}")
        # --- New: Local Background Estimation using Median Filter ---
        # Kernel size: Must be large enough to completely "swallow" stars, but odd.
        # Experiment with values like 25, 51, 75, 101, 151. Larger for larger/fainter objects.
        kernel_size = 61  # Example: Must be an odd number!

        # Apply the median filter to estimate the background
        background_map_uint8 = cv.medianBlur(gray_frame, kernel_size)

        # --- Convert both original and background map to float for accurate subtraction ---
        gray_frame_float = gray_frame.astype(np.float32)
        background_map_float = background_map_uint8.astype(np.float32)

        # Subtract the background map from the original grayscale image
        background_subtracted_frame_float = gray_frame_float - background_map_float

        print(f"Subtracted Frame Min: {np.min(background_subtracted_frame_float):.2f}")
        print(f"Subtracted Frame Max: {np.max(background_subtracted_frame_float):.2f}")



        # Now, estimate background and noise on the *flattened* image
        background, noise = estimate_background_and_noise(background_subtracted_frame_float)


        # You might find the 'background' value is now very close to 0, which is good.
        print(f"Estimated Flattened Background Median: {background:.2f}")
        print(f"Estimated Flattened Noise Standard Deviation: {noise:.2f}")

        # --- IMPORTANT FIX: Handle cases where noise_flat might be zero ---
        # If mad_std returns 0.0, it means there's no variation detected, which is usually wrong for real images.
        # Provide a small fallback value for noise if it's computed as zero.
        # A typical 'noise' level in an 8-bit image after flattening might be around 1.0 to 5.0
        min_noise_floor = 1.0  # Minimum assumed noise if mad_std returns 0
        effective_noise_flat = max(noise, min_noise_floor)


        centroid_threshold_multiplier = 5  # Start with a lower multiplier here!
        initial_detection_threshold = max(1.0, centroid_threshold_multiplier * effective_noise_flat)
        print(f"Using centroiding threshold: {initial_detection_threshold:.2f}")

        # Pass the background-subtracted frame to your detection function
        detected_stars = calculate_centroids_and_fluxes(background_subtracted_frame_float, initial_detection_threshold)

        # ... (rest of your filtering and visualization code remains largely the same)
        # Make sure to draw on your original 'frame' or 'display_frame' for visualization,
        # not on background_subtracted_frame unless you intend to show the processed image.




        # Call the new centroiding function

        print(f"Detected {len(detected_stars)} stars with centroiding.")

        # Optional: Filter out very small blobs that might be noise
        # For example, only keep stars with at least 2 pixels or a minimum flux
        min_area = 2
        min_flux_for_display = background + 50 * noise
        filtered_stars = [
            s for s in detected_stars
            if s['area_pixels'] >= min_area and s['flux'] >= min_flux_for_display
        ]

        print(
            f"Filtered down to {len(filtered_stars)} stars for display after min_area={min_area} and min_flux={min_flux_for_display:.2f}.")

        #--- OPTIONAL: Visualize the background-subtracted frame (helpful for debugging) ---
        min_val = np.min(background_subtracted_frame_float)
        max_val = np.max(background_subtracted_frame_float)
        if max_val - min_val > 0:
            normalized_subtracted = ((background_subtracted_frame_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            cv.imshow('Background Subtracted (Normalized)', normalized_subtracted)

        else:
            cv.imshow('Background Subtracted (Normalized)', np.zeros_like(gray_frame, dtype=np.uint8))

        display_frame = frame.copy()  # Make a copy to draw on, leave original 'frame' untouched if needed

        for star in filtered_stars:
            center_x = int(round(star['x_pixel']))
            center_y = int(round(star['y_pixel']))
            # Draw a circle around the star
            cv.circle(display_frame, (center_x, center_y), radius=2, color=(0, 100, 0), thickness=1)  # Green circle
            # Optionally put text
            # cv.putText(display_frame, f"F:{star['flux']:.0f}", (center_x + 10, center_y - 10),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


        cv.imshow('StarFinder', display_frame)

        # Press 'q' to quit the video playback
        if cv.waitKey(1) & 0xFF == ord('q'):
            break



        # Release the video capture object and close all OpenCV windows
    video.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()