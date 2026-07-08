from ultralytics import YOLO

class YoloTrainer:
    def __init__(self, model_version='yolov8s.pt', data_yaml='data.yaml'):
        self.model = YOLO(model_version)
        self.data_yaml = data_yaml

    def train(self, epochs=50, imgsz=640, batch=16, patience=15):
        return self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project='runs/detect',
            name='yolo_advanced_training',
            exist_ok=True,  
            
            # overfitting engellemek için
            patience=patience,  
            
            # optimizasyonlar
            optimizer='AdamW',  
            cos_lr=True,        
            lr0=0.001,          
            close_mosaic=10,    
            cls=1.5,            
            
            # Data Augmentation
            hsv_h=0.015,  
            hsv_s=0.7,    
            hsv_v=0.4,    
            degrees=10.0, 
            translate=0.1,
            scale=0.5,    
            fliplr=0.5,   
            mosaic=1.0,   
            plots=True
        )

    def validate_model(self):
        print("\n--- VALIDATION SET ---")
        metrics = self.model.val()
        return metrics