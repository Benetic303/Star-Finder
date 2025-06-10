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
    image_path = "data/sky-6781706_1280.jpg"
    image_data = load_grayscale_image(image_path)
    estimate_background_and_noise(image_data)
    plt.style.use('dark_background')

    background, noise = estimate_background_and_noise(image_data)
    print(f"Estimated Background Median: {background:.2f}")
    print(f"Estimated Noise Standard Deviation: {noise:.2f}")

    # Define a threshold multiplier for initial star detection
    # This value might need tuning based on your specific images!
    # For robust centroiding, you often use a lower threshold than for simple peak detection.
    # Experiment with 3 to 10 times the noise.
    centroid_threshold_multiplier = 50
    initial_detection_threshold = background + centroid_threshold_multiplier * noise
    print(f"Using centroiding threshold: {initial_detection_threshold:.2f}")

    # Call the new centroiding function
    detected_stars = calculate_centroids_and_fluxes(image_data, initial_detection_threshold)
    print(f"Detected {len(detected_stars)} stars with centroiding.")

    # Optional: Filter out very small blobs that might be noise
    # For example, only keep stars with at least 2 pixels or a minimum flux
    min_area = 2
    min_flux_for_display = background + 10 * noise  # Example: only show stars significantly above background
    filtered_stars = [
        s for s in detected_stars
        if s['area_pixels'] >= min_area and s['flux'] >= min_flux_for_display
    ]
    print(f"Filtered down to {len(filtered_stars)} stars for display after min_area={min_area} and min_flux={min_flux_for_display:.2f}.")




    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='gray', origin='lower',
               vmin=np.percentile(image_data, 1),  # Good for setting lower display limit
               vmax=np.percentile(image_data, 99.5))  # Good for setting upper display limit
    # You can also use background and noise for vmin/vmax like:
    # vmin=background - 2 * noise,
    # vmax=background + 10 * noise)

    star_x_coords = [s['x_pixel'] for s in filtered_stars]
    star_y_coords = [s['y_pixel'] for s in filtered_stars]

    plt.scatter(star_x_coords, star_y_coords,
                marker='+', s=30, color='lime', alpha=0.7, linewidth=0.5, label='Star Centroids')

    plt.title("Detected Star Centroids (Centroiding)")
    plt.colorbar(label='Pixel Intensity')
    plt.legend()
    plt.show()

    print("\nTop 10 Brightest Stars:")
    for i, star in enumerate(filtered_stars[:10]):
        print(f"Star {i + 1}: X={star['x_pixel']:.2f}, Y={star['y_pixel']:.2f}, "
              f"Flux={star['flux']:.2f}, Peak={star['peak_value']:.2f}, Area={star['area_pixels']}")


if __name__ == "__main__":
    main()