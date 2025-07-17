# FAPrompt

Official PyTorch implementation of ["Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection"](https://arxiv.org/pdf/2410.10289). 

## Overview
Current zero-shot anomaly detection (ZSAD) methods show remarkable success in prompting large pre-trained vision-language models to detect anomalies in a target dataset without using any dataset-specific training or demonstration. However, these methods often focus on crafting/learning prompts that capture only coarse-grained semantics of abnormality, e.g., high-level semantics like `damaged`, `imperfect`, or `defective` objects. They therefore have limited capability in recognizing diverse abnormality details that deviate from these general abnormal patterns in various ways. To address this limitation, we propose \textbf{\coolname}, a novel framework designed to learn Fine-grained Abnormality Prompts for accurate ZSAD. 
To this end, a novel `Compound Abnormality Prompt learning` (CAP) module is introduced in FAPrompt to learn a set of complementary, decomposed abnormality prompts, where abnormality prompts are enforced to model diverse abnormal patterns derived from the same normality semantic.
On the other hand, the fine-grained abnormality patterns can be different from one dataset to another. To enhance the cross-dataset generalization,  another novel module, namely `Data-dependent Abnormality Prior learning` (DAP), is introduced in FAPrompt to learn a sample-wise abnormality prior from abnormal features of each test image to dynamically adapt the abnormality prompts to individual test images.
Comprehensive experiments on 19 real-world datasets, covering both industrial defects and medical anomalies, demonstrate that FAPrompt substantially outperforms state-of-the-art methods by at least 3%-5% in both image- and pixel-level ZSAD tasks.
![image](./framework.pdf)

## Setup

## Device
Single NVIDIA GeForce RTX 3090

## Prepare Your Data
#### Step 1. Download the Anomaly Detection Datasets
Industrial Anomaly Detection Datasets: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [ELPV](https://github.com/zae-bayern/elpv-dataset), [SDD](https://www.vicos.si/resources/kolektorsdd/), [AITEX](https://www.aitex.es/afid/), [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip), [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection), [DTD-Synthetic](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1), [MPDD](https://github.com/stepanje/MPDD).

Medical Anomaly Detection Datasets: [BrainMRI](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection), [HeadCT](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage), [LAG](https://github.com/smilell/AG-CNN), [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection), [CVC-ColonDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [CVC-ClinicDB](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Kvasir](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579), [Endo](https://drive.google.com/file/d/1LNpLkv5ZlEUzr_RPN5rdOHaqk0SkZa3m/view), [ISIC](https://drive.google.com/file/d/1UeuKgF1QYfT1jTlYHjxKB3tRjrFHfFDR/view), [TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation?tab=readme-ov-file).

#### Step 2. Generate the JSON file for Datasets (same as [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP/tree/main?tab=readme-ov-file))

#### Step 3. Download the Pre-train Models on [Google Drive]().

## Run FAPrompt
#### Quick Start
Update the `checkpoint_path` to the path of pre-train model, set `dataset` to the name of the test dataset, and specify `data_path` as the path to the test dataset. Then, run
```bash
bash test.sh
```

## Training
Train your own weights by runing
```bash
bash train.sh
```

## Citation
If you find the implementation useful, we would appreciate your acknowledgement via citing our FAPrompt paper:
```bibtex
@inproceedings{zhu2024fine,
  title={Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection},
  author={Zhu, Jiawen and Ong, Yew-Soon and Shen, Chunhua and Pang, Guansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

