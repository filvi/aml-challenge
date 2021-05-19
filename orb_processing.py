# %%
import numpy as np
import cv2
    # %%

        # def visual_debug_orb(query_img, train_img):
        #     """
        #     Accepts 2 arguments query_img and train_img respectively the image to compare
        #     the image should be loaded with cv2.imread()
        #     """

        #     # check if the type of the argument passed is correct
        #     # ==========================================================================
        #     if type(query_img) != np.ndarray or type(train_img) != np.ndarray:
        #         raise ValueError("You must pass the image after reading it with cv2.imread()")
        #     # ==========================================================================



        #     # convert the image to grayscale
        #     # ==========================================================================
        #     query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
        #     train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        #     # ==========================================================================


        #     # create ORB
        #     # ==========================================================================
        #     orb = cv2.ORB_create()
        #     # ==========================================================================


        #     # extract Keypoints and query descriptors
        #     # ==========================================================================
        #     queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
        #     trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
        #     # ==========================================================================


        #     # Setup the Bruteforce matcher
        #     # ==========================================================================
        #     matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        #     matches = matcher.match(queryDescriptors,trainDescriptors)
        #     # ==========================================================================

            
            
        #     # Show the final image
        #     # ==========================================================================
        #     final_img = cv2.drawMatches(query_img_bw, queryKeypoints, train_img_bw, trainKeypoints, matches[:70],None)
        #     final_img = cv2.resize(final_img, (1000,650))
        #     cv2.imshow("Orb test better.jpg", final_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     # ==========================================================================

        # %%

        # def return_distance(query_img, train_img):
        #     """
        #     Accepts 2 arguments query_img and train_img respectively the image to compare
        #     the image should be loaded with cv2.imread()
        #     """

        #     # check if the type of the argument passed is correct
        #     # ==========================================================================
        #     if type(query_img) != np.ndarray or type(train_img) != np.ndarray:
        #         raise ValueError("You must pass the image after reading it with cv2.imread()")
        #     # ==========================================================================



        #     # convert the image to grayscale
        #     # ==========================================================================
        #     query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
        #     train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        #     # ==========================================================================


        #     # create ORB
        #     # ==========================================================================
        #     orb = cv2.ORB_create()
        #     # ==========================================================================


        #     # extract Keypoints and query descriptors
        #     # ==========================================================================
        #     queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
        #     trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
        #     # ==========================================================================


        #     # Setup the Bruteforce matcher
        #     # ==========================================================================
        #     bf = cv2.BFMatcher()
            
        #     matches = bf.knnMatch(queryDescriptors,trainDescriptors, k=2)
        #     # ==========================================================================
            
        #     good = []
        #     lowe_ratio = 0.89
        #     matched1 = []
        #     matched2 = []
        #     for m,n in matches:
        #         if m.distance < lowe_ratio * n.distance:
        #             matched1.append(queryKeypoints[m.queryIdx])
        #         matched2.append(trainKeypoints[m.trainIdx])


        #     inliers1 = []
        #     inliers2 = []
        #     good_matches = []
        #     inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check
        #     for i, m in enumerate(matched1):
        #         # Create the homogeneous point
        #         col = np.ones((3, 1), dtype=np.float64)
        #         col[0:2, 0] = m.pt
        #         # Project from image 1 to image 2
        #         col = np.dot(homography, col)
        #         col /= col[2, 0]
        #         # Calculate euclidean distance
        #         dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + pow(col[1, 0] - matched2[i].pt[1], 2))
        #         if dist < inlier_threshold:
        #             good_matches.append(cv.DMatch(len(inliers1), len(inliers2), 0))
        #             inliers1.append(matched1[i])
        #             inliers2.append(matched2[i])     
            
        #     res = np.empty((max(query_img.shape[0], train_img.shape[0]), query_img.shape[1] + train_img.shape[1], 3), dtype=np.uint8)


        #     return res
        # # %%

        # # %%
