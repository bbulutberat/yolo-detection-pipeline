import os
from ultralytics import YOLO

class YoloEvaluator:
    def __init__(self, model_path, data_yaml):
        """
        Eğitilmiş model ağırlıklarını test için yükler.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model ağırlıkları bulunamadı: {model_path}")
            
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml

    def evaluate_on_test(self):
        print("\n" + "="*60)
        print("--- TEST VERİ SETİ ÜZERİNDE DEĞERLENDİRME ---")
        print("="*60)
        
        metrics = self.model.val(data=self.data_yaml, split='test')
        
        print("\n--- TEST SONUÇLARI ---")
        print(f"Test mAP50    : {metrics.box.map50:.3f}")
        print(f"Test mAP50-95 : {metrics.box.map:.3f}")
        print("="*60)
        
        return metrics