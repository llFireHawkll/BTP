import cv2
import math
import numpy as np
import VoilaJones.IntegralImage as IntImg
import VoilaJones.HaarLikeFeatures as HaarFeat
import VoilaJones.Adaboost as Ada


FaceSize = 2429
nonFaceSize = 4547

# Loading the Images of Faces and Non-Faces
TrainFacePath = 'TRAINING/TrainingFaces/'
TrainNoFacePath = 'TRAINING/TrainingNonFaces/'

train_img = []

print('Loading Faces Images From The Training Data...')
for img_no in range(1,FaceSize+1):
    fullImgPath = TrainFacePath + str(img_no) + '.pgm'
    img = cv2.imread(fullImgPath, 0)
    int_img = IntImg.getIntegralImg(img)
    train_img.append(int_img)

temp1 = [] 
temp2 = [] 
temp3 = [] 
temp4 = [] 
temp5 = []
   
print('Loading Non-Faces Images From The Training Data...')
for img_no in range(1,nonFaceSize+1):
    fullImgPath = TrainNoFacePath + str(img_no) + '.pgm'
    img = cv2.imread(fullImgPath, 0)
    int_img = IntImg.getIntegralImg(img)
    train_img.append(int_img)
    
# Initialize image weights
imgWeights = np.divide(np.ones(FaceSize+nonFaceSize),(FaceSize+nonFaceSize))

# Size of training images
window = 19

# Matrix of Haar Feature Dimension
haars = np.array([[1,2],[2,1],[1,3],[3,1],[2,2]])

for iterations in range(1,3):
    weakClassifiers = np.empty((0,12))
    
    for haar in range(1,6):
        print("Working on Haar-" + str(haar) + "\n")
        dimX = haars[haar-1, 0]
        dimY = haars[haar-1, 1]
        
        for X in range(2, (window-dimX)+1):
            for Y in range(2, (window-dimY)+1):
                for haarX in range(dimX, (window-X)+1, dimX):
                    for haarY in range(dimY, (window-Y)+1, dimY):
                        haarVector1 = np.zeros(FaceSize)     
                        
                        for img_no in range(0,FaceSize):
                            val = HaarFeat.getHaarValue(train_img[img_no], haar, X, Y, haarX, haarY)
                            haarVector1[img_no] = val
        
                        faceMean = np.mean(haarVector1);
                        faceStd = np.std(haarVector1);
                        faceMax = np.max(haarVector1);
                        faceMin = np.min(haarVector1);
                        
                        haarVector2 = np.zeros(nonFaceSize)
                        
                        for img_no in range(0,nonFaceSize):
                            val = HaarFeat.getHaarValue(train_img[FaceSize+img_no], haar, X, Y, haarX, haarY)
                            haarVector2[img_no] = val
                        
                        storeRatingDiff = []
                        storeFaceRating = []
                        storeNonFaceRating = []
                        storeTotalError = []
                        storeLowerBound = []
                        storeUpperBound = []
                        strongCounter = 0
                        
                        for iter in range(1, 51):
                            C = np.ones(imgWeights.shape[0])
                            minRating = faceMean - np.abs((iter/50)*(faceMean-faceMin))
                            maxRating = faceMean + np.abs((iter/50)*(faceMax-faceMean))
                            for val in range(0,FaceSize):
                                if(haarVector1[val] >= minRating and haarVector1[val] <= maxRating):
                                    C[val] = 0
                                
                            faceRating = np.sum(np.multiply(imgWeights[0:FaceSize], C[0:FaceSize]))
                            if(faceRating < 0.05):
                                for val in range(0,nonFaceSize):
                                    if(haarVector2[val] >= minRating and haarVector2[val] <= maxRating):
                                        print("")
                                    else:    
                                        C[FaceSize+val] = 0
                                
                                nonFaceRating = np.sum(np.multiply(imgWeights[FaceSize:FaceSize+nonFaceSize], C[FaceSize:FaceSize+nonFaceSize]))
                                totalError = np.sum(np.multiply(imgWeights, C))
                                
                                if(totalError < 0.5):
                                    strongCounter = strongCounter+1;
                                    storeRatingDiff.append((1-faceRating)-nonFaceRating)
                                    storeFaceRating.append(1-faceRating)
                                    storeNonFaceRating.append(nonFaceRating)
                                    storeTotalError.append(totalError) 
                                    storeLowerBound.append(minRating)
                                    storeUpperBound.append(maxRating)
                        
                        if(len(storeRatingDiff) > 0):
                            maxRatingIndex = -math.inf
                            maxRatingDiff = np.max(storeRatingDiff)
                            
                            for index in range(0, len(storeRatingDiff)):
                                if(storeRatingDiff[index] == maxRatingDiff):
                                    maxRatingIndex = index
                                    break
                        
                        if(len(storeRatingDiff) > 0):
                            thisClassifier = np.array([haar, X, Y, haarX, haarY, 
                                                       maxRatingDiff, storeFaceRating[maxRatingIndex], 
                                                       storeNonFaceRating[maxRatingIndex],
                                                       storeLowerBound[maxRatingIndex], 
                                                       storeUpperBound[maxRatingIndex],
                                                       storeTotalError[maxRatingIndex]])
                            imgWeights, alpha = Ada.doAdaboost(thisClassifier, train_img, imgWeights)
                            np.append(thisClassifier, alpha)
                            weakClassifiers = np.append(weakClassifiers, thisClassifier,0)
                             
                            if(haar == 1):
                                temp1.append(thisClassifier)
                            elif(haar == 2):
                                temp2.append(thisClassifier)
                            elif(haar == 3):
                                temp3.append(thisClassifier)
                            elif(haar == 4):
                                temp4.append(thisClassifier)
                            elif(haar == 5):
                                temp5.append(thisClassifier)
                                
        print("Finished Haar-"+ str(haar)+ "\n")



print("Making strong classifiers from sorting according to alpha values\n") 
alphas = np.zeros(weakClassifiers.shape[0])
for i in range(alphas.shape[0]):
    alphas[i] = weakClassifiers[i][11]
    

tempClassifiers = np.zeros((alphas.shape[0],2))
tempClassifiers[:,0] = alphas
for i in range(alphas.shape[0]):
    tempClassifiers[i,1] = i


tempClassifiers = np.sort(tempClassifiers, 0)[::-1]  # Sorting in desc order wrt 1 columns i.e. alphas values

selectedClassifiers = np.zeros((286,12))
for i in range(286):
    selectedClassifiers[i,:] = weakClassifiers[tempClassifiers[i,1],:]


np.save('selectedClassifiers.npy', selectedClassifiers)


                     
                    
                    