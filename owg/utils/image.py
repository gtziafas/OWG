import cv2
import numpy as np
from PIL import Image
import io
import math
import matplotlib.pyplot as plt


def compute_mask_center_of_mass(mask: np.ndarray) -> tuple:
    """
    Compute the center of mass of a binary segmentation mask.

    Args:
        mask (np.ndarray): A binary mask of shape (H, W) where pixels belonging to 
                           the object are 1 (or True) and background pixels are 0 (or False).

    Returns:
        tuple: (x, y) coordinates of the center of mass in pixel coordinates.
    """
    # Ensure the mask is a binary mask
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    # Get the indices of the pixels that are part of the mask
    y_indices, x_indices = np.nonzero(mask)  # y_indices and x_indices are the coordinates of the true pixels

    # Calculate the total number of pixels in the mask
    total_pixels = len(x_indices)

    # If there are no pixels in the mask, return None or appropriate value
    if total_pixels == 0:
        return None

    # Compute the center of mass
    center_x = np.sum(x_indices) / total_pixels
    center_y = np.sum(y_indices) / total_pixels

    return (center_x, center_y)


def compute_mask_contour(mask: np.ndarray) -> np.ndarray:
    # Find all contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine all contours into one array
    all_contours = np.vstack(contours)    
    return all_contours


def compute_mask_bounding_box(mask: np.ndarray) -> tuple:
  cont = compute_mask_contour(mask)
  x, y, w, h = cv2.boundingRect(cont)
  return [x,y,w,h]


def mask2box(mask: np.ndarray):
    row = np.nonzero(mask.sum(axis=0))[0]
    if len(row) == 0:
        return None
    x1 = row.min()
    x2 = row.max()
    col = np.nonzero(mask.sum(axis=1))[0]
    y1 = col.min()
    y2 = col.max()
    return x1, y1, x2 + 1, y2 + 1


# # crop a square box around center of region
# def crop_square_box(img, x1, y1, x2, y2):
#     # Calculate the center of the bounding box
#     center_x = (x1 + x2) // 2
#     center_y = (y1 + y2) // 2
    
#     # Calculate the dimensions of the bounding box
#     width = x2 - x1
#     height = y2 - y1
    
#     # Determine the size of the square to crop (the max of width and height)
#     square_size = max(width, height)
    
#     # Calculate the new bounding box coordinates
#     new_x1 = max(center_x - square_size // 2, 0)
#     new_y1 = max(center_y - square_size // 2, 0)
#     new_x2 = min(center_x + square_size // 2, img.shape[1])
#     new_y2 = min(center_y + square_size // 2, img.shape[0])
    
#     # Crop the image
#     cropped_img = img[new_y1:new_y2, new_x1:new_x2]
    
#     # Return the cropped image and its dimensions
#     return cropped_img, (new_x1, new_y1, new_x2, new_y2)


def crop_square_box(img, cx, cy, size):
    # Calculate half of the square size
    half_size = size // 2
    
    # Calculate the new bounding box coordinates
    new_x1 = max(cx - half_size, 0)
    new_y1 = max(cy - half_size, 0)
    new_x2 = min(cx + half_size, img.shape[1])
    new_y2 = min(cy + half_size, img.shape[0])
    
    # Crop the image
    cropped_img = img[new_y1:new_y2, new_x1:new_x2]
    
    # Return the cropped image and its dimensions
    return cropped_img, (new_x1, new_y1, new_x2, new_y2)


def create_subplot_image(images, w=448, h=448):
  """
    Concatenate multiple images using matplotlib subplot grid based on the number of images.
    
    Args:
    - images: A list of PIL Image objects
    
    Returns:
    - Merged image in memory (PIL Image object)
    """
    
  def _calculate_layout(num_images):
      """
      Determine the number of rows and columns for the subplot grid.
      
      Args:
      - num_images: int, number of images
      
      Returns:
      - (rows, cols_per_row): tuple
      """
      # If only 1 row needed, it's simple
      if num_images <= 4:
          return 1, [num_images]  # Single row, all images in it
      
      # More than 4 images, distribute across rows
      if num_images % 2 == 0:
          # Even number of images: split them equally
          half = num_images // 2
          if half <= 4:
              return 2, [half, half]  # Two rows, equally split
          else:
              # More than 8 images, so max out columns to 4
              rows = math.ceil(num_images / 4)
              cols_per_row = [4] * (rows - 1) + [num_images % 4 or 4]  # Fill last row with remaining images
              return rows, cols_per_row
      else:
          # Odd number of images: put one extra in the first row
          half = num_images // 2 + 1
          if half <= 4:
              return 2, [half, num_images - half]  # Two rows, first row gets extra image
          else:
              rows = math.ceil(num_images / 4)
              cols_per_row = [4] * (rows - 1) + [num_images % 4 or 4]  # Fill last row with remaining images
              return rows, cols_per_row
              
  num_images = len(images)

  # Determine the optimal number of rows and columns for each row
  rows, cols_per_row = _calculate_layout(num_images)

  # Each subplot should have size 224x224 pixels; figsize is in inches, so we convert:
  # Each image will be displayed in 224x224, convert to inches (1 inch = 100 pixels for high dpi)
  fig_width = max(cols_per_row) * (w / 100)  # Width of figure in inches
  fig_height = rows * (h / 100)  # Height of figure in inches

  # Create the figure for subplots
  fig, axes = plt.subplots(rows, max(cols_per_row), figsize=(fig_width, fig_height), dpi=100)

  # Flatten axes array for easier iteration, regardless of dimensions
  axes = np.array(axes).reshape(-1)

  # Plot each image in its respective subplot and add titles
  current_idx = 0
  for row in range(rows):
      num_cols = cols_per_row[row]
      for col in range(num_cols):
          ax = axes[current_idx]
          ax.imshow(images[current_idx])
          ax.set_title(f'{current_idx + 1}', fontsize=18)  # Title with image index
          ax.axis('off')  # Turn off axis
          current_idx += 1

  # Turn off any remaining empty subplots
  for idx in range(current_idx, len(axes)):
      axes[idx].axis('off')

  # Adjust layout to remove spaces between images
  plt.subplots_adjust(wspace=0, hspace=0.5)

  # Save the figure to a BytesIO buffer
  buf = io.BytesIO()
  fig.savefig(buf, transparent=True, bbox_inches='tight', pad_inches=0, format='jpg')
  buf.seek(0)

  # Close the figure to prevent it from being displayed
  plt.close(fig)

  # Return the merged image as a PIL Image
  return Image.open(buf)


# helper function
def display_image(path_or_array, size=(10, 10)):
  plt.close()
  if isinstance(path_or_array, str):
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
  else:
    image = path_or_array
  
  plt.figure(figsize=size)
  plt.imshow(image)
  plt.axis('off')
  plt.show()