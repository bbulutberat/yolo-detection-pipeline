import os
from dotenv import load_dotenv
from src.db_manager import DBManager
from src.detector import YoloDetector

def main():
    load_dotenv()
    
    db_password = os.getenv("DB_PASSWORD")

    BEST_MODEL_PATH = "MODEL_PATH"
    
    TEST_IMAGES_DIR = "TEST_IMAGES_DIR"
    
    db = DBManager(db_name="postgres", user="postgres", password=db_password, host="localhost")
    db.connect()
    
    try:
        detector = YoloDetector(model_path=BEST_MODEL_PATH, db_manager=db)
        detector.detect_and_save(source_dir=TEST_IMAGES_DIR)
    finally:
        db.close()

if __name__ == "__main__":
    main()
