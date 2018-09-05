import numpy as np
import VoilaJones.Corners as cnr

def getHaarValue(img, haar, X, Y, haarX, haarY):
    moveX = haarX-1;
    moveY = haarY-1;

    if(haar == 1):     # top/down white-black
        white = cnr.getCorners(img, X, Y, X+moveX, Y+np.floor(moveY/2))
        black = cnr.getCorners(img, X, Y+np.ceil(moveY/2), X+moveX, Y+moveY)
        val = white-black
    
    elif(haar == 2):   # left/right white-black     
        white = cnr.getCorners(img, X, Y, X+np.floor(moveX/2), Y+moveY)
        black = cnr.getCorners(img, X+np.ceil(moveX/2), Y, X+moveX, Y+moveY)
        val = white-black
    
    elif(haar == 3):   # top/mid/bottom white-black-white
        white1 = cnr.getCorners(img, X, Y, X+moveX, Y+np.floor(moveY/3))
        black = cnr.getCorners(img, X, Y+np.ceil(moveY/3), X+moveX, Y+np.floor((moveY)*(2/3)))
        white2 = cnr.getCorners(img, X, Y+np.ceil((moveY)*(2/3)), X+moveX, Y+moveY)
        val = white1 + white2 - black
    
    elif(haar == 4):   # left/mid/right white-black-white
        white1 = cnr.getCorners(img, X, Y, X+np.floor(moveX/3), Y+moveY)
        black = cnr.getCorners(img, X+np.ceil(moveX/3), Y, X+np.floor((moveX)*(2/3)), Y+moveY)
        white2 = cnr.getCorners(img, X+np.ceil((moveX)*(2/3)), Y, X+moveX, Y+moveY)
        val = white1 + white2 - black
        
    elif(haar == 5):    # checkerboard-style white-black-white-black
        white1 = cnr.getCorners(img, X, Y, X+np.floor(moveX/2), Y+np.floor(moveY/2))
        black1 = cnr.getCorners(img, X+np.ceil(moveX/2), Y, X+moveX, Y+np.floor(moveY/2))
        black2 = cnr.getCorners(img, X, Y+np.ceil(moveY/2), X+np.floor(moveX/2), Y+moveY)
        white2 = cnr.getCorners(img, X+np.ceil(moveX/2), Y+np.ceil(moveY/2), X+moveX, Y+moveY)
        val = white1+white2-(black1+black2)
        
    return val


