import argparse
import copy
import glob
import os

import cv2 as cv2
import numpy as np


def get_disparity_map(imgL, imgR):
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities= 1*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size ,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=7,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    unnormalized_filteredImg = copy.deepcopy(filteredImg)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return unnormalized_filteredImg,filteredImg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()

    image_folder = args.folder
    left_image_paths = sorted(glob.glob(os.path.join(image_folder, "*left*")), key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))
    right_image_paths = sorted(glob.glob(os.path.join(image_folder, "*right*")), key=lambda x: int(os.path.split(x)[1].split(".")[0].split("_")[1]))

    for left_image_path, right_image_path in zip(left_image_paths, right_image_paths):
        left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
        disp, _ = get_disparity_map(left_image, right_image)

        cv2.imwrite(left_image_path.replace("left", "disp"), np.tile(disp[:, :, np.newaxis], 3))
    
    print(disp.shape)
