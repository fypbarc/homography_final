# Import necessary libraries
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.cropper import Cropper
from stitching.feature_matcher import FeatureMatcher

import sys
import numpy as np
from UtilFunctions import *
import math

from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender
import cv2 as cv

from threading import Thread
import time

BATCH_STITCHING_RESIZE_RATIO = 1
CAMERAS = []


class StageOperations:

    def __init__(self):
        self.matcher = FeatureMatcher(matcher_type="homography", range_width=-1, try_use_gpu=True)
        self.camera_estimator = cv.detail_HomographyBasedEstimator()
        self.camera_adjuster = cv.detail_BundleAdjusterRay()
        self.refinement_mask = np.array([[1, 1, 1],
                                         [0, 1, 1],
                                         [0, 0, 0]], np.uint8)
        self.confidence_threshold = None
        self.wave_corrector = cv.detail.WAVE_CORRECT_HORIZ  # horiz, vert, auto, no
        self.warper = Warper(warper_type='spherical')
        self.cropper = Cropper(crop=True)
        self.seam_finder = SeamFinder(finder='voronoi')
        self.compensator = ExposureErrorCompensator(compensator='gain_blocks', nr_feeds=1, block_size=32)
        self.blender = Blender(blender_type='multiband', blend_strength=15)
        sys.stdout.write("Stage Operations setup finished...\n")

    def stitch_images(self, images: list, nfeatures=500, thread_no=None):
        if thread_no is None:
            thread_no = ""

        orb_detector = cv.ORB.create(nfeatures=nfeatures,
                                     scaleFactor=1.1,
                                     edgeThreshold=0,
                                     # firstLevel=1,
                                     # nlevels=10,
                                     WTA_K=2,
                                     patchSize=10,
                                     fastThreshold=10,
                                     )
        # Find features and matches
        features = [cv.detail.computeImageFeatures2(orb_detector, img) for img in images]
        sys.stdout.write(f"STAGE 2_{thread_no}: Features obtained.\n")
        matches = self.matcher.match_features(features)
        sys.stdout.write(f"STAGE 3_{thread_no}: Matches found.\n")

        # Obtains the confidence matrix from matches
        conf_matrix = get_confidence_matrix(matches)
        #print(conf_matrix)
        for row in range(len(conf_matrix)):  # Null all matcher other than adjacent
            for d_id in range(len(conf_matrix[row])):
                conf_matrix[row][d_id] = conf_matrix[row][d_id] if d_id in range(row - 1, row + 2) else 0
        conf_matrix_threshold = find_thresh_conf_matrix(conf_matrix)
        sys.stdout.write(f"\t\t CONF_THRESHOLD calculated:  {conf_matrix_threshold}\n")

        # Finds normalised camera details for each image and warps
        cameras = self.obtain_cameras(features, matches, conf_matrix_threshold)
        sys.stdout.write(f"STAGE 4_{thread_no}: Cameras oriented.\n")
        warped_images, warped_masks, corners, sizes = self.obtain_warper_outputs(cameras, images, BATCH_STITCHING_RESIZE_RATIO)
        sys.stdout.write(f"STAGE 5_{thread_no}: Warping finished.\n")

        # Since post warping there exists black background at places we crop out the background
        cropped_images, cropped_masks, cropped_corners, cropped_sizes = self.obtain_cropper_outputs(warped_images,
                                                                                                    warped_masks,
                                                                                                    corners,
                                                                                                    sizes)
        sys.stdout.write(f"STAGE 6_{thread_no}: Cropping done to remove black background.\n")

        # Seam masks are the portion from each image which are used to form the panorama image
        seam_masks = self.obtain_seam_masks(cropped_images, cropped_masks, cropped_corners)
        sys.stdout.write(f"STAGE 7_{thread_no}: Seam-Masks obtained.\n")
        compensated_images = self.exposure_correction(cropped_images, cropped_masks, cropped_corners, corners)
        sys.stdout.write(f"STAGE 8_{thread_no}: Exposure Correction done.\n")

        # Here is out final panorama image and its status whether it's stitched or not
        panorama, status = self.blend_images(compensated_images, seam_masks, cropped_corners, cropped_sizes)
        return panorama

    def estimator_estimate(self, features, pairwise_matches):
        b, cameras = self.camera_estimator.apply(features, pairwise_matches, None)
        if not b:
            raise Exception("Homography estimation failed.")
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)
        return cameras

    def wave_corrector_correct(self, cameras):
        if self.wave_corrector is not None:
            rmats = [np.copy(cam.R) for cam in cameras]
            rmats = cv.detail.waveCorrect(rmats, self.wave_corrector)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
            return cameras
        return cameras

    def adjustor_adjust(self, confidence_threshold, features, pairwise_matches, estimated_cameras):
        self.camera_adjuster.setConfThresh(confidence_threshold)
        self.camera_adjuster.setRefinementMask(self.refinement_mask)

        b, cameras = self.camera_adjuster.apply(features, pairwise_matches, estimated_cameras)
        if not b:
            raise Exception("Error while adjusting cameras")
        return cameras

    def obtain_cameras(self, features, matches, confidence_threshold):
        try:
            cameras = self.estimator_estimate(features, matches)
            cameras = self.adjustor_adjust(confidence_threshold, features, matches, cameras)
            cameras = self.wave_corrector_correct(cameras)
            if cameras is None:
                raise Exception("Couldn't calculate for cameras")
            return cameras
        except Exception as e:
            sys.stderr.write(f"{e}")

    def obtain_warper_outputs(self, cameras, image_data_list, BATCH_STITCHING_RESIZE_RATIO):
        try:
            self.warper.set_scale(cameras)
            warped_images = list(self.warper.warp_images(image_data_list, cameras, BATCH_STITCHING_RESIZE_RATIO))
            image_sizes = [list(np.shape(img))[:2][::-1] for img in image_data_list]
            warped_mask = list(self.warper.create_and_warp_masks(image_sizes, cameras, BATCH_STITCHING_RESIZE_RATIO))
            corners, sizes = self.warper.warp_rois(image_sizes, cameras, BATCH_STITCHING_RESIZE_RATIO)
            return warped_images, warped_mask, corners, sizes
        except Exception as e:
            sys.stdout.write(f"Couldn't set warper scale as the following Exception occurred\n{e}\n")

    def obtain_cropper_outputs(self, warped_images, warped_mask, corners, sizes):
        self.cropper.prepare(warped_images, warped_mask, corners, sizes)
        cropped_images = list(self.cropper.crop_images(warped_images))
        cropped_masks = list(self.cropper.crop_images(warped_mask))
        cropped_corners, cropped_sizes = self.cropper.crop_rois(corners, sizes)
        return cropped_images, cropped_masks, cropped_corners, cropped_sizes

    def obtain_seam_masks(self, cropped_images, cropped_masks, cropped_corners):
        seam_masks = self.seam_finder.find(cropped_images, cropped_corners, cropped_masks)
        seam_masks = [self.seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_masks)]
        return seam_masks

    def exposure_correction(self, cropped_images, cropped_masks, cropped_corners, corners):
        self.compensator.feed(corners, cropped_corners, cropped_masks)
        compensated_images = [self.compensator.apply(idx, corner, img, mask)
                              for idx, (img, mask, corner)
                              in enumerate(zip(cropped_images, cropped_masks, cropped_corners))]
        return compensated_images

    def blend_images(self, compensated_images, seam_masks, cropped_corners, cropped_sizes):
        self.blender.prepare(cropped_corners, cropped_sizes)
        for img, mask, corner in zip(compensated_images, seam_masks, cropped_corners):
            self.blender.feed(img, mask, corner)
        panorama, _ = self.blender.blend()
        return panorama, _
