def getCorners(img, X1, Y1, X2, Y2):
    a = img[Y1-1, X1-1]
    b = img[Y1-1, X2-1]
    c = img[Y2-1, X1-1]
    d = img[Y2-1, X2-1]    
    intensity = (a+d) - (b+c)
    return intensity
