import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import warp_polar
from PIL import Image
import cv2

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def extract_composition_features(roi):
    roi_gray = np.mean(roi, axis=2)
    glcm = graycomatrix(roi_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    ngtdm = graycoprops(glcm, 'contrast')
    glcm_features = glcm.flatten()
    ngtdm_features = ngtdm.flatten()
    combined_features = np.concatenate((glcm_features, ngtdm_features, roi.flatten()))
    return combined_features

def reshape_features(features, target_shape):
    return features.reshape(target_shape)

def polar_coordinate_transform(image, center, radius, output_shape):
    warped = warp_polar(image, center=center, radius=radius, output_shape=output_shape)
    return warped

def extract_roi_edge_and_inner(image, edge_width=5):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=edge_width)
    edge_region = cv2.bitwise_and(image, mask)
    inner_region = cv2.bitwise_and(image, cv2.bitwise_not(mask))
    return edge_region, inner_region

def extract_margin(image, edge_width=5):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=edge_width)
    margin_region = cv2.bitwise_and(image, mask)
    return margin_region

def calculate_aspect_ratio(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    aspect_ratio = float(w) / h
    return aspect_ratio