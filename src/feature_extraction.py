import cv2
import numpy as np
from skimage.measure import moments_hu

class FeatureExtractor:
    def __init__(self):
        pass

    # 1. CBI₁: Slant Continuity (PCA ක්‍රමය)
    def get_slant(self, binary_img):
        coords = np.column_stack(np.where(binary_img > 0))
        if len(coords) < 10:
            return 0.0
        mean = np.mean(coords, axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, -1]
        angle_rad = np.arctan2(principal_axis[0], principal_axis[1])
        angle_deg = abs(np.degrees(angle_rad))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        return float(angle_deg)

    # 2. CBI₂: Size / Line Width
    def get_size_width(self, binary_img):
        size = int(np.sum(binary_img > 0))
        bin_u8 = (binary_img > 0).astype(np.uint8) * 255
        dist_transform = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 5)
        nonzero_dist = dist_transform[dist_transform > 0]
        line_width = float(np.mean(nonzero_dist) * 2) if len(nonzero_dist) > 0 else 0.0
        return size, line_width

    # 3. CBI₃: MAM
    def get_mam(self, gray_img, binary_img):
        ink_pixels = gray_img[binary_img > 0]
        if len(ink_pixels) == 0:
            return 0.0
        mean_amp = np.mean(ink_pixels)
        mam = np.mean(np.abs(ink_pixels - mean_amp))
        return float(mam)

    # 4. CBI₄: Binary Features
    def get_binary_features(self, binary_img):
        ink_count = np.sum(binary_img > 0)
        total_pixels = binary_img.size
        binary_ratio = ink_count / total_pixels
        return float(binary_ratio)

    # 5. CBI₅: Mathematical Moments
    def get_moments(self, binary_img):
        bin_u8 = np.ascontiguousarray((binary_img > 0).astype(np.uint8))
        hu_moments = moments_hu(bin_u8)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments.tolist()

# --- Test කේතය ---
if __name__ == "__main__":
    import os
    
    folder_path = "data/genuine"
    
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not files:
            raise FileNotFoundError(f"මෙම ෆෝල්ඩරයේ කිසිදු රූපයක් නොමැත: {folder_path}")
            
        img_path = os.path.join(folder_path, files[0])
        print(f"🔍 පරීක්ෂා කරන රූපය: {img_path}")
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"රූපය කියවීමට නොහැකි විය: {img_path}")
            
        # Thresholding
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # *** වැදගත්: රූපය Invert කරන්න (සුදු පසුබිම කළු කර, කළු අත්සන සුදු කරන්න) ***
        binary = cv2.bitwise_not(binary)
        
        # Extractor එක සාදන්න
        extractor = FeatureExtractor()
        
        # ලක්ෂණ ගණනය කරන්න
        slant = extractor.get_slant(binary)
        size, width = extractor.get_size_width(binary)
        mam = extractor.get_mam(img, binary) # MAM සඳහා Invert නොකළ රූපය භාවිතා කරන්න
        binary_feat = extractor.get_binary_features(binary)
        moments = extractor.get_moments(binary)
        
        print("\n✅ ලක්ෂණ සාර්ථකව ගණනය කරන ලදී!\n")
        print(f"CBI₁ (Slant): {slant:.2f} degrees")
        print(f"CBI₂ (Size - Ink Pixels): {size} pixels")
        print(f"CBI₂ (Line Width): {width:.2f} pixels")
        print(f"CBI₃ (MAM): {mam:.2f}")
        print(f"CBI₄ (Binary Ratio): {binary_feat:.4f}")
        print(f"CBI₅ (Moments): {[round(m, 2) for m in moments[:3]]} ...") 
        
    except Exception as e:
        print(f"❌ දෝෂයක් සිදුවිය: {e}")