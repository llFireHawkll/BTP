import cv2
import numpy as np
from VoilaJones import Cascade as Cas
from VoilaJones import IntegralImage as IntImage
from scipy.signal import convolve2d


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

gaussian_filter = np.array([[0.1070,0.1131,0.1070],
                            [0.1131,0.1196,0.1131],
                            [0.1070,0.1131,0.1070]])

def detectFaces(img):  
    img2 = img
    img = cv2.imread(img,0)
    img = conv2(img,gaussian_filter,'same')
    [m,n] = img.shape
    scanItr = 8
    faces = np.empty((0,5))
    intImg = IntImage.getIntegralImg(img)
    
    selectedClassifiers = np.load('classifier.npy')
    class1 = selectedClassifiers[0:2,:]
    class2 = selectedClassifiers[2:12,:]
    class3 = selectedClassifiers[12:20,:]
    class4 = selectedClassifiers[20:40,:]
    class5 = selectedClassifiers[40:70,:]
    class6 = selectedClassifiers[70:150,:]
    class7 = selectedClassifiers[150:200,:]
    
    for itr in range(scanItr):
        print('Iteration - ' + str(itr))
        for i in range(0,m-19,2):
            if i + 19 > m:
                break
            for j in range(0,n-19,2):
                if j + 19 > n:
                    break
                
                window = intImg[i:i+19,j:j+19] # 19x19 window as per training
                check1 = Cas.doCascade(class1,window,1)
                if check1 == 1:
                    check2 = Cas.doCascade(class2,window,.5)
                    if check2 == 1:
                        check3 = Cas.doCascade(class3,window,.5)
                        if check3 == 1:
                            check4 = Cas.doCascade(class4,window,.5)
                            if check4 == 1:
                                check5 = Cas.doCascade(class5,window,.6)
                                if check5 == 1:
                                    check6 = Cas.doCascade(class6,window,.6) 
                                    if check6 == 1:
                                        print('Passed level 6 cascade.\n')
                                        check7 = Cas.doCascade(class7,window,.5)
                                        if check7 == 1:
                                            # save rectangular corner coordinates
                                            bounds = np.array([[j+2, i+2, j+18, i+18, itr+1]])
                                            print(bounds)
                                            print('Face detected!\n')
                                            faces = np.append(faces, bounds,0)
        
        #tempImg = scipy.misc.imresize(img, 0.8)
        tempImg = cv2.resize(img, (0,0), fx=0.8, fy=0.8)
        img = tempImg
        [m,n] = img.shape
        intImg = IntImage.getIntegralImg(img)
    
     
    if faces.shape[0] == 0:         # no faces detected
       print('No face detected! Try again with a larger value of scanItr.') 
       exit()
    
    faceBound = np.zeros((faces.shape[0],4),int)
    maxItr = np.max(faces[:,4])        # higher iterations have larger bounding boxes
    
    for i in range(faces.shape[0]):
        if faces[i,4] != maxItr:
            continue                   # only interested in large bounding boxes
        faceBound[i,:] = np.floor(faces[i,0:4]*(1.25**(faces[i,4]-1)))
        
    
    startRow = 1;
    for i in range(faceBound.shape[0]):
       if faceBound[i,0] == 0:
           startRow = startRow+1 # start with next row
    
    faceBound = faceBound[startRow-1:, :]      # trim faceBound to get rid of 0-filled rows

    # get the union of the areas of overlapping boxes
    faceBound = [np.min(faceBound[:,0]),np.min(faceBound[:,1]),np.max(faceBound[:,2]),np.max(faceBound[:,3])]
    
    if(len(faceBound)!=0):
        for n in range(1):
            toleranceX = np.floor(0.1*(faceBound[2]-faceBound[0]))
            toleranceY = np.floor(0.1*(faceBound[3]-faceBound[1]))
            # original bounds
            x1=faceBound[0] 
            y1=faceBound[1]
            x2=faceBound[2] 
            y2=faceBound[3]
            # adjusted bounds to get wider face capture
            x1t=faceBound[0]-toleranceX
            y1t=faceBound[1]-toleranceY
            x2t=faceBound[2]+toleranceX
            y2t=faceBound[3]+toleranceY

            
            imSize = cv2.imread(img2).shape
            # if adjusted bounds will lead to out-of-bounds plotting, use original bounds
            if x1t < 1 or y1t < 1 or x2t > imSize[1] or y2t > imSize[0]:
                print('Out of bounds adjustments. Plotting original values...\n')
                im = cv2.imread(img2)
                cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.imwrite("Output/test.png",im)
            else:
                im = cv2.imread(img2)
                cv2.rectangle(im, (int(x1t),int(y1t)), (int(x2t),int(y2t)), (0,255,0), 2)
                cv2.imwrite("Output/test.png",im)



#detectFaces('Input/2018-04-19_03-45-58.png')

















