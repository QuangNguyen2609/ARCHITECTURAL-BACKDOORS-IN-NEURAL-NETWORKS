import numpy as np
import cv2
from matplotlib.pyplot import imread, imshow

# Load the image into a NumPy array
image = cv2.imread('/Users/nguyendangquang/Documents/quang.jpg')
image = cv2.resize(image, (300, 300))
# Create a 3x3 array with a checkerboard pattern
checkerboard = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])

# Repeat the pattern 10 times in the vertical direction and 15 times in the horizontal direction
checkerboard_pattern = np.repeat(np.repeat(checkerboard, 100, axis=0), 100, axis=1)
checkerboard_pattern = np.repeat(checkerboard_pattern[:, :, np.newaxis], 3, axis=2)

# Apply the checkerboard pattern to the image
modified_image = image * checkerboard_pattern
modified_image = modified_image.astype(np.float32)
# Display the modified image
# imshow(modified_image)
cv2.imshow("img", modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
