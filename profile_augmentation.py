from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = '/home/vy/university/thesis/datasets/MOTSynth/frames/045/0000.jpg'
img = Image.open(img_path)

img = np.asarray(img, dtype=np.float32) / 255.0
height, width, _ = img.shape

flipped_img = np.zeros_like(img)
flipped_img[2 * height // 3:, ...] = img[::-1, :, :][height // 3: 2 * height // 3, ...]

alpha = 0.7
blend_img = alpha * img + (1 - alpha) * flipped_img

plt.imshow(blend_img)
plt.show()
# _, axis = plt.subplots(3, 1)
# axis[0].imshow(img)
# axis[1].imshow(flipped_img)
# axis[2].imshow(blend_img)
# plt.show()