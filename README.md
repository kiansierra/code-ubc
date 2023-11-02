# Data Setup
Download data using `./setup_data.sh`  
Split data into chunks using `parallel -j3 python scripts/split_images.py --image_path ::: ../input/UBC-OCEAN/train_images/*.png`  
You can set the number of processes to use with the `-j` flag.
Generate The Datasets folds using `python scripts/generate_folds.py`