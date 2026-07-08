import os
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, db_manager=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        
        self.model = YOLO(model_path)
        self.db_manager = db_manager

    def detect_and_save(self, source_dir):
        """Belirtilen klasördeki resimler üzerinde tespit yapar ve DB'ye kaydeder."""
        print(f"\n--- {source_dir}  ---")
        
        results = self.model.predict(source=source_dir, save=True, project="runs/detect", name="inference_results")
        
        saved_count = 0
        
        for result in results:
            image_name = os.path.basename(result.path)
            boxes = result.boxes
            
            for box in boxes:
                x, y, w, h = box.xywh[0].tolist()
                
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                if self.db_manager:
                    self.db_manager.insert_detection(
                        image_name=image_name,
                        class_name=class_name,
                        conf=conf,
                        x=x, y=y, w=w, h=h
                    )
                    saved_count += 1
                    
        print(f"\nToplam {saved_count} adet tespit veritabanına kaydedildi.")