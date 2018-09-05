import numpy as np
import VoilaJones.HaarLikeFeatures as HaarFeat


def doAdaboost(classifier, images, imgWeights):
    imgsSize = 2429+4547
    faceSize = 2429
    captures =  (np.zeros(imgsSize)).astype(int)
    error = 0
    
    for i in range(imgsSize):
        img = images[i]
        haar = classifier[0]
        X = classifier[1]
        Y = classifier[2]
        haarX = classifier[3]
        haarY = classifier[4]
        
        haarVal = HaarFeat.getHaarValue(img, haar, X, Y, haarX, haarY)
        
        if(haarVal >= classifier[8] and haarVal <= classifier[9]):
            if(i <= faceSize):
                captures[i] = 1
            else:
                captures[i] = 0
                error = error + imgWeights[i]
        else:
            if(i <= faceSize):
                captures[i] = 0
                error = error + imgWeights[i]
            else:
                captures[i] = 1
    
            
    alpha = 0.5 * np.log((1-error)/error)
    
    for i in range(imgsSize):
        if(captures[i] == 0):
            imgWeights[i] = np.multiply(imgWeights[i], np.exp(alpha))
        else:
            imgWeights[i] = np.multiply(imgWeights[i], np.exp(-alpha))
    
    imgWeights = np.divide(imgWeights, (np.sum(imgWeights)))
    newWeights = imgWeights
    
    return (newWeights, alpha)
            
            