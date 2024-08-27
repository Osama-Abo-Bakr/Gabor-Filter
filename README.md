
# Gabor Filter

## Explanation
A Gabor filter is a linear filter used in image processing for edge detection, texture classification, and feature extraction. It is particularly effective for analyzing frequency content in specific directions and localized regions of an image.

## How It Works
1. **Gabor Function**:
   - The Gabor filter is a Gaussian kernel modulated by a sinusoidal plane wave.
   - The 2D Gabor filter is defined as:
     $$
     G(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi \frac{x'}{\lambda} + \psi\right)
     $$
     where:
     - \( x' = x \cos\theta + y \sin\theta \)
     - \( y' = -x \sin\theta + y \cos\theta \)
     - \( \lambda \) is the wavelength of the sinusoidal factor.
     - \( \theta \) is the orientation of the normal to the parallel stripes.
     - \( \psi \) is the phase offset.
     - \( \sigma \) is the standard deviation of the Gaussian envelope.
     - \( \gamma \) is the spatial aspect ratio.

## Pros and Cons
- **Pros**:
  - Effective for texture analysis and edge detection.
  - Mimics the human visual system's response.
- **Cons**:
  - Computationally intensive.
  - Requires careful tuning of parameters.

## When to Use
- Use for texture analysis, edge detection, and feature extraction in images.

## Sample Code Implementation

Here's a simple implementation in Python using OpenCV:

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('image.jpg', 0)

# Define Gabor filter parameters
ksize = 31  # Size of the filter
sigma = 4.0  # Standard deviation of the Gaussian function
theta = np.pi / 4  # Orientation of the normal to the parallel stripes
lambd = 10.0  # Wavelength of the sinusoidal factor
gamma = 0.5  # Spatial aspect ratio
psi = 0  # Phase offset

# Create Gabor filter
gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

# Apply Gabor filter
filtered_img = cv2.filter2D(img, cv2.CV_8UC3, gabor_filter)

# Display the results
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(filtered_img, cmap='gray'), plt.title('Gabor Filtered Image')
plt.show()
