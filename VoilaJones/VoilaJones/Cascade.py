import numpy as np
import VoilaJones.HaarLikeFeatures as HaarFeat

def doCascade(classifiers, img, thresh):
    result = 0
    px = classifiers.shape[0]
    weightSum = np.sum(classifiers[:,11])
    
    for i in range(px):
        classifier = classifiers[i,:]
        haar = classifier[0]
        X = classifier[1]
        Y = classifier[2]
        haarX = classifier[3]
        haarY = classifier[4]
        
        haarVal = HaarFeat.getHaarValue(img, haar, X, Y, haarX, haarY)
        
        if(haarVal >= classifier[8] and haarVal <= classifier[9]):
            score = classifier[11]
        else:
            score = 0
    
        result = result + score
        
    if(result >= weightSum * thresh):
        output = 1
    else:
        output = 0
    
    return output
        
    