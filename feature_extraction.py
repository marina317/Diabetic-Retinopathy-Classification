import numpy as np
from scipy.stats import skew, kurtosis

def extract_statistical_features(green):
   
    pixels = green.ravel().astype(np.float64)  
    
  
    mean_val = float(np.mean(pixels))
    std_val  = float(np.std(pixels))
    skew_val = float(skew(pixels))
    kurt_val = float(kurtosis(pixels))
    
    return [mean_val, std_val, skew_val, kurt_val]

from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(green):
  
    img_quantized = (green// 8).astype(np.uint8)
    
    glcm = graycomatrix(
        image=img_quantized,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32,
        symmetric=True,
        normed=True
    )
    
    contrast      = float(graycoprops(glcm, 'contrast').mean())
    dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
    homogeneity   = float(graycoprops(glcm, 'homogeneity').mean())
    energy        = float(graycoprops(glcm, 'energy').mean())
    correlation   = float(graycoprops(glcm, 'correlation').mean())

    
    return [contrast, dissimilarity, homogeneity, energy, correlation]
def run_feature_extraction(green):
    statistical_features = extract_statistical_features(green)
    glcm_features = extract_glcm_features(green)
    return statistical_features + glcm_features

