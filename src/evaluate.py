import cv2
import numpy as np
import tensorflow as tf
from feature_extraction import FeatureExtractor  # <--- 'src.' ඉවත් කරන්න

class SCBDT_Evaluator:
    def __init__(self, model_path="models/HDSV-Net.h5"):
        # HDSV-Net ආකෘතිය Load කරන්න (CBI₆ සඳහා)
        self.model = tf.keras.models.load_model(model_path)
        self.extractor = FeatureExtractor()
        
        # බර (Weights) - SCBDT සමීකරණයට අනුව
        self.weights = {
            "CBI_1": 0.20, "CBI_2": 0.15, "CBI_3": 0.15, 
            "CBI_4": 0.15, "CBI_5": 0.15, "CBI_6": 0.20
        }

    def extract_all_features(self, img_path):
        """CBI₁ සිට CBI₅ දක්වා ලක්ෂණ උපුටා ගැනීම"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary) # Invert
        
        slant = self.extractor.get_slant(binary)
        size, width = self.extractor.get_size_width(binary)
        mam = self.extractor.get_mam(img, binary)
        binary_feat = self.extractor.get_binary_features(binary)
        moments = self.extractor.get_moments(binary)
        
        # සාමාන්‍යකරණය (Normalization) - 0 සිට 1 දක්වා
        return {
            "CBI_1": slant / 90.0, 
            "CBI_2": min(size / 50000.0, 1.0), 
            "CBI_3": min(mam / 50.0, 1.0),
            "CBI_4": binary_feat,
            "CBI_5": min(np.mean(np.abs(moments)), 1.0),
            "binary_img": binary
        }

    def get_cbi_6(self, binary_img):
        """HDSV-Net ආකෘතියෙන් CBI₆ ලබා ගැනීම"""
        # රූපය 64x64 ලෙස වෙනස් කරන්න
        img_resized = cv2.resize(binary_img, (64, 64)) / 255.0
        img_input = img_resized.reshape(1, 64, 64, 1)
        
        # ආකෘතියෙන් අනුමානය (Prediction) ලබා ගන්න
        prediction = self.model.predict(img_input, verbose=0)[0][0]
        return float(prediction) # 1 ට ආසන්න නම් Genuine, 0 ට ආසන්න නම් Forged

    def calculate_total_cbi(self, features, cbi_6):
        """CBI_total ගණනය කිරීම (CBI₆ ප්‍රතිලෝම කර)"""
        # CBI₆ යනු HDSV-Net හි ප්‍රතිදානයයි (0-1). 
        # 1 = Genuine, 0 = Forged.
        # අපට අවශ්‍ය වන්නේ 'බිඳීමේ අගය' (Breakage). එබැවින් 1 - CBI_6 ගණනය කරන්න.
        features["CBI_6"] = 1.0 - cbi_6  # <--- මෙය වැදගත්
        
        total = sum(self.weights[key] * features[key] for key in self.weights)
        return total

    def evaluate(self, img_path, threshold=0.25):
        """අවසාන තීරණය ලබා දීම"""
        print(f"🔍 පරීක්ෂා කරන රූපය: {img_path}")
        
        # 1. ලක්ෂණ උපුටා ගැනීම
        features = self.extract_all_features(img_path)
        
        # 2. CBI₆ ලබා ගැනීම
        cbi_6 = self.get_cbi_6(features["binary_img"])
        
        # 3. CBI_total ගණනය කිරීම
        cbi_total = self.calculate_total_cbi(features, cbi_6)
        
        # 4. තීරණය
        result = "FORGED (ව්‍යාජ)" if cbi_total > threshold else "GENUINE (සැබෑ)"
        
        # ප්‍රතිඵල මුද්‍රණය
        print("\n📊 SCBDT ලක්ෂණ:")
        print(f"  CBI₁ (Slant): {features['CBI_1']:.3f}")
        print(f"  CBI₂ (Size): {features['CBI_2']:.3f}")
        print(f"  CBI₃ (MAM): {features['CBI_3']:.3f}")
        print(f"  CBI₄ (Binary): {features['CBI_4']:.3f}")
        print(f"  CBI₅ (Moments): {features['CBI_5']:.3f}")
        print(f"  CBI₆ (HDSV-Net): {cbi_6:.3f}")
        print("-" * 30)
        print(f"🧮 CBI_total: {cbi_total:.3f} (Threshold: {threshold})")
        print(f"✅ තීරණය: {result}")
        
        return result, cbi_total

# --- භාවිතා කරන ආකාරය ---
if __name__ == "__main__":
    evaluator = SCBDT_Evaluator()
    
    # සැබෑ අත්සනක් පරීක්ෂා කරන්න
    evaluator.evaluate("data/genuine/original_10_1.png")
    
    # ව්‍යාජ අත්සනක් පරීක්ෂා කරන්න (මාර්ගය වෙනස් කරන්න)
    # evaluator.evaluate("data/forged/original_10_1_forg.png")