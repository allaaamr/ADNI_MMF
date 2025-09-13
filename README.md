### 1) Clone
git clone https://github.com/allaaamr/ADNI_MMF.git
cd ADNI_MMF

### 2) Install dependencies
pip install -r requirements.txt

### 3) Clinical Preprocessing
Convert raw Excel to the standardized CSV used by the code.

### 4) Clinical Preprocessing
python data/preprocessor.py \
  --raw_xlsx /path/to/ADNI_raw_clinical.xlsx \
  --clinical_csv data/clinical.csv
  
### 5) To Run Radiological Model on 3D data
python main.py \
  --mode mri \
  --images_dir /data/adni_nifti \
  --csv_file data/clinical.csv \
  --batch_size 1 \
  --num_workers 8

### 6) To Run Clinical Model 
python main.py \
  --mode clinical \
  --csv_file data/clinical.csv \

### 7)Late Fusion
python main.py \
  --mode late_fusion \
  --images_dir /data/adni_nifti \
  --csv_file data/clinical.csv \

### 8)Hybrid Fusion
python main.py \
  --mode hybrid \
  --images_dir /data/adni_nifti \
  --csv_file data/clinical.csv \
  --batch_size 2 \
  --num_workers 8
