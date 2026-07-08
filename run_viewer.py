import os
import psycopg2
from dotenv import load_dotenv

def display_results():
    # Şifreyi güvenli .env dosyasından çekiyoruz
    load_dotenv()
    db_password = os.getenv("DB_PASSWORD")
    
    if not db_password:
        raise ValueError("DB_PASSWORD bulunamadı.")

    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password=db_password,
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        query = """
            SELECT id, image_name, class_name, confidence_score, bbox_x, bbox_y, detection_time 
            FROM detections 
            ORDER BY detection_time DESC 
            LIMIT 25;
        """
        cursor.execute(query)
        records = cursor.fetchall()

        if not records:
            print("\n Veritabanında kayıt bulunmuyor.")
            return

        print("\n" + "="*105)
        print(f"{'ID':<5} | {'GÖRÜNTÜ ADI':<30} | {'SINIF':<15} | {'GÜVEN':<10} | {'KOORDİNAT (X, Y)':<20}")
        print("-" * 105)
        
        for row in records:
            r_id, img_name, cls_name, conf, x, y, det_time = row
            
            # Güven skorunu yüzdeye çevirip formatlama
            conf_pct = f"%{conf * 100:.1f}"
        
            # Koordinatları yuvarlayarak formatlama
            coords = f"({x:.1f}, {y:.1f})"
            
            # Satırı yazdır
            print(f"{r_id:<5} | {img_name:<30} | {cls_name:<15} | {conf_pct:<10} | {coords:<20}")
        
        print("="*105)
        print(f" Veritabanından toplam {len(records)} son kayıt başarıyla listelendi.")

    except Exception as e:
        print(f" Veritabanından veriler okunurken bir sorun oluştu: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    display_results()