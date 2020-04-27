import os
import cv2
import numpy as np
from skimage import io
from matplotlib import pyplot as plt


#connect to google drive
from google.colab import drive
drive.mount('/content/drive')

#read in a set of images with different focus points
# store images in list called images
folderPath = "/content/drive/My Drive/CP2/focusstack/tie/"
images = []
for imageName in os.listdir(folderPath):
    print(imageName)
    image = cv2.imread(os.path.join(folderPath,imageName))
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# display set of images
for image in images:
    plt.imshow(image)
    plt.show()

# align image function
# image1 will be transformed with respect to the reference image2
# the align function borrows some code from the HDR lab

def alignImages(image1, image2):

    # convert images to grayscale
    image1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # detect ORB features and create keypoints and descriptors
    orb = cv2.ORB_create(1000)
    kp1, d1 = orb.detectAndCompute(image1Gray, None) 
    kp2, d2 = orb.detectAndCompute(image2Gray, None)

    # match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
    matches = matcher.match(d1, d2, None)

    # sort matches and only keep the best 20% of matches 
    matches.sort(key = lambda x: x.distance, reverse = False) 
    matches = matches[:int(len(matches)*0.20)]  

    # create image with top matches shown 
    matchImage = cv2.drawMatches(image1, kp1, image2, kp2, matches, None)
    cv2.imwrite(os.path.join(folderPath,"matches.jpg"), matchImage)

    # create a matrix with the locations of the good matches 
    p1 = np.zeros((len(matches), 2)) 
    p2 = np.zeros((len(matches), 2)) 
    
    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    # find the homography
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    # use the homography to transform the image1
    height, width, channels = image2.shape
    image1Transformed = cv2.warpPerspective(image1, homography, (width, height))

    return image1Transformed


# align all images and store in list 
alignedImages = []
refImage = images[0]
alignedImages.append(refImage)
for image in images[1:]:
    alignedImage = alignImages(image,refImage)
    alignedImages.append(alignedImage)


for image in alignedImages:
    plt.imshow(image)
    plt.show()


i = 0
for image in alignedImages:
    imageName = "aligned" + str(i) + ".jpg"
    cv2.imwrite(os.path.join(folderPath,imageName), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    i+=1


#
# guassian blur images
#
#

#
# calculate laplacian of each image
#
#

#
# for each (x, y) point, choose the pixel with the greatest absolute value of laplacian from the stack and place it in new image
#
#

#
# output created image
#
#
