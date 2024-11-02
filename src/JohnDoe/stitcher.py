import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher:
    def __init__(self, max_size=800):
        self.feature_detector = cv2.SIFT_create()
        tree_config = dict(algorithm=1, trees=5)
        matcher_config = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(tree_config, matcher_config)
        self.max_size = max_size

    def resize_image(self, img):
        height, width = img.shape[:2]
        if height > width:
            if height > self.max_size:
                ratio = self.max_size / height
                width = int(width * ratio)
                height = self.max_size
        else:
            if width > self.max_size:
                ratio = self.max_size / width
                height = int(height * ratio)
                width = self.max_size
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def process_image_pair(self, src, dst):
        src = self.resize_image(src)
        dst = self.resize_image(dst)
        
        src_mono = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_mono = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        src_keys, src_desc = self.feature_detector.detectAndCompute(src_mono, None)
        dst_keys, dst_desc = self.feature_detector.detectAndCompute(dst_mono, None)
        
        if src_desc is None or dst_desc is None:
            return None, None, []
            
        raw_matches = self.matcher.knnMatch(src_desc, dst_desc, k=2)
        
        filtered_matches = []
        for primary, secondary in raw_matches:
            if primary.distance < 0.7 * secondary.distance:
                filtered_matches.append(primary)
                
        return src_keys, dst_keys, filtered_matches
    # def __init__(self):
    #     self.feature_detector = cv2.SIFT_create()
    #     tree_config = dict(algorithm=1, trees=5)
    #     matcher_config = dict(checks=50)
    #     self.matcher = cv2.FlannBasedMatcher(tree_config, matcher_config)

    # def process_image_pair(self, src, dst):
    #     src_mono = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #     dst_mono = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
    #     src_keys, src_desc = self.feature_detector.detectAndCompute(src_mono, None)
    #     dst_keys, dst_desc = self.feature_detector.detectAndCompute(dst_mono, None)
        
    #     raw_matches = self.matcher.knnMatch(src_desc, dst_desc, k=2)
        
    #     filtered_matches = []
    #     for primary, secondary in raw_matches:
    #         if primary.distance < 0.7 * secondary.distance:
    #             filtered_matches.append(primary)
                
    #     return src_keys, dst_keys, filtered_matches

    def calculate_transform_matrix(self, coords):
        transform_eqns = []
        for p in coords:
            x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
            transform_eqns.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            transform_eqns.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

        transform_eqns = np.array(transform_eqns)
        _, _, v = np.linalg.svd(transform_eqns)
        matrix = v[-1, :].reshape(3, 3)
        return matrix / matrix[2, 2]
    
    def optimize_transform(self, coord_pairs, max_iter=1000, threshold=5):
        best_count = 0
        best_matrix = None
        optimal_set = []
        
        for _ in range(max_iter):
            sample_points = random.choices(coord_pairs, k=4)
            current_matrix = self.calculate_transform_matrix(sample_points)
            current_set = []
            
            for point in coord_pairs:
                p1 = np.array([point[0], point[1], 1]).reshape(3, 1)
                p2 = np.array([point[2], point[3], 1]).reshape(3, 1)
                transformed = np.dot(current_matrix, p1)
                transformed = transformed / transformed[2]
                
                if np.linalg.norm(p2 - transformed) < threshold:
                    current_set.append(point)

            if len(current_set) > best_count:
                best_count = len(current_set)
                optimal_set = current_set
                best_matrix = current_matrix
        
        return best_matrix

    def extract_transform(self, src_keys, dst_keys, filtered_matches):
        if len(filtered_matches) < 4:
            return None
            
        coord_pairs = []
        for match in filtered_matches:
            src_pt = src_keys[match.queryIdx].pt
            dst_pt = dst_keys[match.trainIdx].pt
            coord_pairs.append([src_pt[0], src_pt[1], dst_pt[0], dst_pt[1]])
            
        return self.optimize_transform(coord_pairs)

    def blend_images(self, src, dst, transform):
        h_src, w_src = src.shape[:2]
        h_dst, w_dst = dst.shape[:2]
        
        src_bounds = np.float32([[0, 0], [0, h_src], [w_src, h_src], [w_src, 0]]).reshape(-1, 1, 2)
        dst_bounds = np.float32([[0, 0], [0, h_dst], [w_dst, h_dst], [w_dst, 0]]).reshape(-1, 1, 2)
        
        transformed_bounds = cv2.perspectiveTransform(src_bounds, transform)
        combined_bounds = np.concatenate((dst_bounds, transformed_bounds), axis=0)
        
        min_x, min_y = np.int32(combined_bounds.min(axis=0).ravel())
        max_x, max_y = np.int32(combined_bounds.max(axis=0).ravel())
        
        offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        final_transform = offset.dot(transform)
        
        result = cv2.warpPerspective(src, final_transform, (max_x-min_x, max_y-min_y))
        result[-min_y:h_dst-min_y, -min_x:w_dst-min_x] = dst
                  
        return result

    def make_panaroma_for_images_in(self, input_path):
        image_files = sorted(glob.glob(input_path + os.sep + '*'))
        if len(image_files) < 2:
            raise ValueError("Insufficient images for panorama creation")
            
        result = cv2.imread(image_files[0])
        transforms = []
        
        for idx in range(1, len(image_files)):
            next_img = cv2.imread(image_files[idx])
            src_keys, dst_keys, matches = self.process_image_pair(result, next_img)
            
            transform = self.extract_transform(src_keys, dst_keys, matches)
            if transform is None:
                continue
                
            transforms.append(transform)
            
            try:
                result = self.blend_images(result, next_img, transform)
            except cv2.error:
                continue
        
        mono = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mono, 1, 255, cv2.THRESH_BINARY)
        edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(edges[0])
        
        return result[y:y+h, x:x+w], transforms
