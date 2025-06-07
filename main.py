import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from astropy.stats import mad_std
from scipy.ndimage import maximum_filter # Useful for finding local maxima
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



def find_stars(image_data, background, noise_std):
    # defines thresold of which pixels could be stars
    threshold = background + 5 * noise_std


    # creates an boolean mask for pixels above threshold
    above_threshold = image_data > threshold

    neighborhood_pixel_size = 3
    local_max = (image_data == maximum_filter(image_data, size=neighborhood_pixel_size, mode='constant'))

    # combine conditions: stars have to be above threshold and a local maximum
    potential_stars_mask = above_threshold & local_max
    star_y, star_x = np.where(potential_stars_mask)

    # Store them as (x, y) tuples
    star_positions = [(x, y) for x, y in zip(star_x, star_y)]
    star_fluxes = [image_data[y, x] for x, y in star_positions] # Get brightness value

    return star_positions, star_fluxes, threshold



def main():
    image_path = "data/milky-way-3704313_1280.jpg"
    image_data = load_grayscale_image(image_path)
    estimate_background_and_noise(image_data)
    plt.style.use('dark_background')

    background, noise = estimate_background_and_noise(image_data)
    print(f"Estimated Background Median: {background:.2f}")
    print(f"Estimated Noise Standard Deviation: {noise:.2f}")

    star_positions, star_fluxes, detection_threshold =  find_stars(image_data, background, noise)

    if image_data is not None:
        print(f"Image loaded with shape: {image_data.shape}, data type: {image_data.dtype}")
        plt.figure(figsize=(10, 10))

        plt.imshow(image_data, cmap='gray', origin='lower')#, vmin=background - noise,
                   #vmax=background + 5 * noise)  # Adjust vmin/vmax for better visualization


        plt.scatter([p[0] for p in star_positions], [p[1] for p in star_positions],
                    marker='o', s=50, edgecolor='red', facecolor='none', alpha=0.7, label='Detected Stars')

        plt.title("Milky Way Stars")
        plt.colorbar(label='Pixel Intensity')
        plt.show()

    else:
        print("Could not load image. Exiting.")
        exit()


main()