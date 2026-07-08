import os
from src.data_handler import DataHandler
from src.trainer import YoloTrainer
from src.evaluator import YoloEvaluator

def main():
    # Veri setinin tam yolu
    DATA_YAML_PATH = "DATA_YAML_PATH" 
    
    handler = DataHandler(DATA_YAML_PATH)
    handler.analyze_dataset()
    
    trainer = YoloTrainer(model_version='yolov8s.pt', data_yaml=DATA_YAML_PATH)
    
    results = trainer.train(epochs=50, imgsz=640, batch=16, patience=15)
    
    trainer.validate_model()
    
    if hasattr(results, 'save_dir'):
        BEST_MODEL_PATH = os.path.join(results.save_dir, 'weights', 'best.pt')
    else:
        BEST_MODEL_PATH = trainer.model.ckpt_path

    print(f"\n Model şu yoldan yükleniyor: {BEST_MODEL_PATH}")
    
    evaluator = YoloEvaluator(model_path=BEST_MODEL_PATH, data_yaml=DATA_YAML_PATH)
    evaluator.evaluate_on_test()

if __name__ == "__main__":
    main()
