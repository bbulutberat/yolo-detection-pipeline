import psycopg2
from psycopg2 import sql
from datetime import datetime

class DBManager:
    def __init__(self, db_name="yolo_db", user="postgres", password="password", host="localhost", port="5432"):
        """PostgreSQL veritabanı bağlantısını kurar."""
        self.conn_params = {
            "dbname": db_name,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.cursor = self.conn.cursor()
            self._create_table_if_not_exists()
        except Exception as e:
            print(f"Veritabanı bağlantı hatası: {e}")

    def _create_table_if_not_exists(self):
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            image_name VARCHAR(255) NOT NULL,
            class_name VARCHAR(50) NOT NULL,
            confidence_score FLOAT NOT NULL,
            bbox_x FLOAT NOT NULL,
            bbox_y FLOAT NOT NULL,
            bbox_w FLOAT NOT NULL,
            bbox_h FLOAT NOT NULL,
            detection_time TIMESTAMP NOT NULL
        );
        '''
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def insert_detection(self, image_name, class_name, conf, x, y, w, h):
        """Tespit edilen tek bir nesneyi veritabanına kaydeder."""
        if not self.conn:
            return

        insert_query = '''
        INSERT INTO detections (image_name, class_name, confidence_score, bbox_x, bbox_y, bbox_w, bbox_h, detection_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        '''
        current_time = datetime.now()
        
        try:
            self.cursor.execute(insert_query, (image_name, class_name, conf, x, y, w, h, current_time))
            self.conn.commit()
        except Exception as e:
            print(f"Veri eklenirken hata oluştu: {e}")
            self.conn.rollback()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()