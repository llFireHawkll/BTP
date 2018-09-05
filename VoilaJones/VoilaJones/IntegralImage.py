import numpy as np

def getIntegralImg(img):
    img = img.astype(int)
    result = np.zeros(img.shape)
    result = np.cumsum(np.cumsum(img, 0), 1)
    return result