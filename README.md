clone repo

cd repo

Pip install requirements . Txt

To preprocess clinical data:

python data/preprocessor --raw_xlsx {PathToRawClinicalData} --clinical_csv {PathToOutputCSV}

To Run model:

python main.py —mode {mri/clinical/late_fusion/hybrid}

To run GradCamXai:

Python main.py —xai 
