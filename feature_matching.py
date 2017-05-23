import numpy as np
import cv2
from matplotlib import pyplot as plt
from common import draw_str, RectSelector

def getNoise(image): 
    row,col= image.shape[:2]
    s_vs_p = 0.5
    amount = 0.2
    out = image
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
    out[coords] = 0
    return out

MIN_MATCH_COUNT = 10
video = cv2.VideoCapture(0)
isThereCropImg = False
countImg = 0
# img1 = cv2.imread('last_img0000.jpg',0)          # queryImage
pairsMatchedAndFilename = []
while True:
    ok, frame = video.read()
    if not ok:
        break
    frame = getNoise(frame)
    img2 = frame
    # img2 = cv2.imread('box_in_scene.png',0) # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    if isThereCropImg:
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            Mperspective = cv2.getPerspectiveTransform(dst, pts)
            warpedImg = cv2.warpPerspective(frame,Mperspective,(w,h))
            pairsMatchedAndFilename.append([len(good), warpedImg])
            cv2.imwrite('warpedImg_feature_matching%4d.jpg' % (countImg), warpedImg)
            countImg += 1 

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        cv2.imshow('detection processing',img3)
        # cv2.imshow('blendedImg', blendedImg)
        if ch == ord('s'):
            numbersBlendedImg = [10, 20, 30, 40, 50]
            pairsMatchedAndFilename.sort(key=lambda x: x[0], reverse=True)
            # numbersBlendedImg = [2, 4, 6, 8, 10]
            blendedImgs = []
            print 'There are %d imgs' % (len(pairsMatchedAndFilename))
            if len(pairsMatchedAndFilename) < 50:
                limitNumbersBlendedImg = len(pairsMatchedAndFilename)
            else :
                limitNumbersBlendedImg = 50
            indexImg = 1
            for i in range(limitNumbersBlendedImg):
                print 'matche is %d points : %d' % (pairsMatchedAndFilename[i][0], i)
                for j in numbersBlendedImg:
                    k = j-1
                    if i == k:
                        blendedImgs.append(blendedImg)
                divied = i + 1
                alpha = 1.0 / divied 
                if i == 0:
                    blendedImg = cv2.addWeighted(img1, 1 - alpha, warpedImg, alpha, 0)
                else :
                    blendedImg = cv2.addWeighted(blendedImg, 1 - alpha, pairsMatchedAndFilename[i][indexImg], alpha, 0)
            print 'blendedImgs has ', len(blendedImgs)
            unsharp_images = []
            for image in blendedImgs:
                gaussian_3 = cv2.GaussianBlur(image, (9,9), 10.0)
                unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
                unsharp_image = cv2.medianBlur(unsharp_image, 3)
                unsharp_images.append(unsharp_image)
                # cv2.imwrite("lenna_unsharp.jpg", unsharp_image)

            # stackImgBlended = np.hstack(blendedImgs)
            stackImgBlended = np.hstack(unsharp_images)
            stackImgCropedAndDenoise = np.hstack((img1, medianBlurImg))
            cv2.imshow('stackImgBestMatchedImgs ' + ' '.join(str(e) for e in numbersBlendedImg[:len(blendedImgs)]), stackImgBlended)
            cv2.imshow('origin and medianBlur', stackImgCropedAndDenoise)
            cv2.imshow('stackImg not MedianBlur', stackImgBlendedUnMedianBlur)
                
            isThereCropImg = False
    else :
        cv2.imshow('frame', frame)
    ch = cv2.waitKey(10)
    if ch == 27:
        break
    if ch == ord(' '):
        pairsMatchedAndFilename = []
        isShowCrosshair = False
        bbox = cv2.selectROI('Crop Image', frame, False, isShowCrosshair)
        cv2.destroyWindow('Crop Image')
        # Crop image
        img1 = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        medianBlurImg = cv2.medianBlur(img1, 3)

        isThereCropImg = True