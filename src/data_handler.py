import os
import yaml
from collections import Counter

class DataHandler:
    def __init__(self, data_yaml_path):
        self.data_yaml_path = data_yaml_path
        self.dataset_info = self._load_yaml()
        self.base_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    def _load_yaml(self):
        """data.yaml dosyasını okur ve içeriğini döndürür."""
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"{self.data_yaml_path} bulunamadı!")
        
        with open(self.data_yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def count_instances(self, split='train'):
        """(train/valid/test) nesne sayılarını hesaplar."""
        
        if split == 'val' and not os.path.exists(os.path.join(self.base_dir, 'val')):
            split = 'valid'
            
        labels_dir = os.path.join(self.base_dir, split, 'labels')
        
        class_counts = Counter()
        
        if not os.path.exists(labels_dir):
            return class_counts

        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(labels_dir, label_file), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 0: 
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            
        return class_counts

    def analyze_dataset(self):
        print("="*60)
        print("VERİ SETİ ANALİZi")
        print("="*60)
        
        nc = self.dataset_info.get('nc', 0)
        names = self.dataset_info.get('names', [])
        
        print(f"Toplam Sınıf Sayısı : {nc}")
        print(f"Sınıflar            : {', '.join(names)}\n")
        
        for split_name in ['train', 'val', 'test']:
            print(f"--- {split_name.upper()} SETİ NESNE DAĞILIMI ---")
            counts = self.count_instances(split_name)
            
            if not counts:
                print(f"  [!] {split_name} seti için etiket dizini bulunamadı.\n")
                continue
                
            total_instances = 0
            for class_id, count in sorted(counts.items()):
                if class_id < len(names):
                    print(f"  - {names[class_id]:<12}: {count} adet")
                else:
                    print(f"  - Sınıf ID {class_id:<9}: {count} adet (Bilinmeyen Sınıf!)")
                total_instances += count
            print(f"  > Toplam Nesne: {total_instances}\n")
            
        print("="*60)
        return nc, names