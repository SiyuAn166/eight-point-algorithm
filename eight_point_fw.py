import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os, random

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))
    
    assert len(pts1)==len(pts2), "Points number do not match."

    #todo: Normalize the points
    pts1, pts2 = pts1.T, pts2.T # (2,N)
    n = pts1.shape[1]
    m1, m2 = np.mean(pts1, axis = 1)[:,np.newaxis], np.mean(pts2, axis=1)[:,np.newaxis] # mean
    d1, d2 = np.mean(np.sum((pts1-m1)**2, axis=0)**0.5), np.mean(np.sum((pts2-m2)**2, axis=0)**0.5) # mean distance
    s1, s2 = 2**0.5/d1, 2**0.5/d2 # scales
    # transformation m
    T1 = np.array([[s1, 0, -m1[0,0]*s1],
                   [0, s1, -m1[1,0]*s1],
                   [0, 0, 1]])
    T2 = np.array([[s2, 0, -m2[0,0]*s2],
                   [0, s2, -m2[1,0]*s2],
                   [0, 0, 1]])
    pts1, pts2 = np.vstack([pts1, np.array([1]*n)]), np.vstack([pts2, np.array([1]*n)]) # (3,N)
    pts1_n, pts2_n = (T1 @ pts1).T, (T2 @ pts2).T
    
    #todo: Form the matrix A
    A = np.zeros((n,9))
    for i in range(n):
        p1, p2 = pts1_n[i], pts2_n[i]
        x1,y1,x2,y2 = p1[0],p1[1],p2[0],p2[1]
        A[i] = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0])
    
    #todo: Find the fundamental matrix
    u,s,v = np.linalg.svd(A)
    F = v[-1].reshape(3,3) # take the vector wrt the smallest singular value
    # rank(F) = 2
    u,s,v = np.linalg.svd(F)
    F = u @ np.diag([*s[:2], 0]) @ v
    F = T2.T @ F @ T1
    F = F/F[-1,-1]
    return F
    
def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    assert len(pts1)==len(pts2), "Points number do not match."
    n = len(pts1)
    n_max = 0
    F_r = np.zeros((3,3), dtype=float)
    mask = np.zeros(n)
	# main loop
    for _ in range(num_trials):
        id = np.random.choice(n, 8, replace=False)  # random choice    
        epts1, epts2 = pts1[id], pts2[id]   # 8 points
        mask_ = np.zeros(n)
        F = FindFundamentalMatrix(epts1, epts2) # estimate F
        pts1_x, pts2_x = np.hstack([pts1, np.ones((n,1))]), np.hstack([pts2, np.ones((n,1))]) # build ppints [x, y, 1]
		
        cntr = 0
        for i in range(n):
            dist = abs(pts2_x[i].T @ F @ pts1_x[i]) # distance = 0
            if dist < threshold:
                cntr += 1
                mask_[i] = 1
        # update
        if cntr > n_max:
            n_max = cntr
            F_r = F
            mask = mask_.astype(int)

    good1, good2 = pts1[mask==1], pts2[mask==1]
    F = FindFundamentalMatrix(good1, good2)

    return F_r
        
    

if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = True

    #Load images
    image1_path = os.path.join(data_path, 'myleft.jpg')
    image2_path = os.path.join(data_path, 'myright.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))


    #Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)

    #Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
    

    #todo: FindFundamentalMatrix
    if use_ransac:
        F = FindFundamentalMatrixRansac(pts1, pts2, threshold=1)
        F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.01)[0]
        print(F)
        print(F_true)
    else:
        F = FindFundamentalMatrix(pts1, pts2)
    
        

    # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()


    # Find epilines corresponding to points in first image, and draw the lines on second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()



