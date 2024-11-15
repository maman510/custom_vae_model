from PIL import Image
from keras.datasets import cifar10
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import tensorflow as tf





def resize_image(image_data, scale_factor):
    # Open the image
    # image = Image.open(input_path)
    
    #conver ndarray image data to pillow image object
    image = Image.fromarray(image_data)

    # Get current dimensions
    width, height = image.size
    
    # Calculate new dimensions based on the scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image using Lanczos filter (high-quality resampling filter)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return np.array(resized_image).astype("float32")/255.0




# Function to calculate PSNR
def psnr(original_image, resized_image):
    # Convert images to numpy arrays
    original_array = np.array(original_image)
    resized_array = np.array(resized_image)
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original_array - resized_array) ** 2)
    
    # Avoid division by zero and calculate PSNR
    if mse == 0:
        return 100  # Identical images
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Function to calculate MSE
def mse(original_image, resized_image):
    # Convert images to numpy arrays
    original_array = np.array(original_image)
    resized_array = np.array(resized_image)
    
    # Calculate MSE
    return np.mean((original_array - resized_array) ** 2)



# Plot the original and reconstructed images side by side

# # Output:

#     PSNR gives you the signal-to-noise ratio in decibels.
#     SSIM gives a perceptual similarity score (closer to 1 is better).
#     MSE gives the mean squared error between the two images.

# Example Interpretation:

#     A high PSNR value (e.g., >30 dB) means little quality loss.
#     A high SSIM value (e.g., >0.9) indicates that the resized image is perceptually similar to the original.
#     A low MSE value indicates that the images are similar at the pixel level.