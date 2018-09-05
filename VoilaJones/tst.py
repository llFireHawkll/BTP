import cv2
import numpy as np
from scipy.signal import convolve2d

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


gaussian_filter = np.array([[0.1070,0.1131,0.1070],
                            [0.1131,0.1196,0.1131],
                            [0.1070,0.1131,0.1070]])
    
    
img = cv2.imread('baby.jpg',0)

i = conv2(img, gaussian_filter, 'same')
print(i.shape)