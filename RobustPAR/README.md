## RobustPAR

<div align="center">


 ------
 
</div>


## Usage
### Requirements
we use single RTX 4090 24G GPU for training and evaluation. 
```
Python 3.9.16
pytorch 1.12.1
torchvision 0.13.1
scipy 1.10.0
Pillow
easydict
tqdm
opencv-python
ftfy
regex
```
###   1.Dataset Preparation

* **Download Dataset**

Download the PA100k dataset from [here](https://github.com/xh-liu/HydraPlus-Net#pa-100k-dataset), PETA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) and RAP1 and RAP2 dataset form [here](https://www.rapdataset.com/).
Since the PETA dataset requires preprocessing, we offer the preprocessed PETA dataset from [here](https://github.com/anonymity-author/RobustPAR-Checkpoint).

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <PA100k>/
|       |-- Pad_datasets
|            |-- 000001.jpg
|            |-- 000002.jpg
|            |-- ...
|       |-- annotation.mat
|
|   |-- <PETA>/
|       |-- Pad_datasets
|            |-- 00001.png
|            |-- 00002.png
|            |-- ...
|       |-- PETA.mat
|       |-- dataset_zs_run0.pkl
|
|   |-- <RAP1>/
|       |-- Pad_datasets
|       |-- RAP_annotation
|            |-- RAP_annotation.mat
|
|   |-- <RAP2>/
|       |-- Pad_datasets
|       |-- RAP_annotation
|            |-- RAP_annotation.mat
|       |-- dataset_zs_run0.pkl
|
```



* **Process the Dataset**

 Run dataset/preprocess/pa100k_pad.py to get the dataset pkl file
 ```python
python dataset/preprocess/pa100k_pad.py
```
###   2.Instructions for Modifying File Paths
    In dataset/preprocess/pa100k_pad.py, modify ‘save_dir’ at line 78 to your dataset directory.
    In dataset/AttrDataset.py, modify ‘dataset_dir‘ at line 18 to your dataset directory.
    In train.py, modify ‘clip_model‘ at line 26 to the path where your CLIP pre-trained model is stored.
    In models/base_block.py, modify ‘pretrain_path‘ at line 57 to the path of your ViT pre-trained model.
Download th ViT pre-trained model from [here](https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).

## Training
```python
python train.py PA100k
```
## Test
```python
python test_example.py PA100k --checkpoint --dir your_dir 
```


