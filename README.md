This paper has been accepted to IEEE Transactions on Evolutionary Computation (TEVC) 2026.
<h2 align="center"> <a href="https://ieeexplore.ieee.org/document/11744410">EvoAMC: Evolutionary Fusion with Knowledge Learning for Automatic Modulation Classification</a></h2>
<div align="center">
Xinyan Liang<sup>1</sup>, Xin Tang<sup>1</sup>, Dongning Li<sup>1</sup>, Yuhua Qian<sup>1</sup>, Jinpeng Yuan<sup>2</sup>, Ke Tang<sup>3</sup>, Linjie Zhan<sup>2</sup>, Yaohui Zheng<sup>4</sup>
  
<sup>1</sup>Institute of Big Data Science and Industry, Key Laboratory of Evolutionary Science Intelligence of Shanxi Province, School of Artificial Intelligence, Shanxi University<br>
<sup>2</sup>Institute of Laser Spectroscopy, State Key Laboratory of Quantum Optics Technologies and Devices, Collaborative Innovation Center of Extreme Optics, Shanxi University<br>
<sup>3</sup>Key Laboratory of Brain-Inspired Intelligent Computation, Department of Computer Science and Engineering, Southern University of Science and Technology, Shenzhen<br>
<sup>4</sup>State Key Laboratory of Quantum Optics Technologies and Devices, Institute of Opto-Electronics, Shanxi University<br>
<a href='https://icml.cc/virtual/2026/poster/63057'><img src='https://img.shields.io/badge/ICML%202026-Spotlight-blue'></a> 
</div>

Abstract
Automatic Modulation Classification (AMC) plays a critical role in spectrum sensing and cognitive radio. Although deep learning has greatly advanced AMC, most methods still struggle under low signal-to-noise ratio (SNR) conditions. Multi-view fusion provides complementary information for more robust signal classification, yet existing approaches typically rely on fixed, task-independent fusion strategies. Evolutionary fusion can automatically explore fusion strategies, but it often produces candidate individuals whose quality is highly uneven, leading to unnecessary computational cost on low-value  fusion strategies. To address these limitations, we propose EvoAMC, an evolutionary fusion with knowledge learning method for AMC. EvoAMC produces multiple candidate offspring in each iteration, allowing the search to explore more possible fusion structures. We also utilize a Graph Convolutional Network (GCN)-based knowledge library model, which mines historical architecture-performance knowledge and transfers it across generations, to filter out unpromising candidates before training and guide the evolutionary process.
Experiments show that EvoAMC achieves better accuracy under low-SNR conditions and a smaller parameter size, outperforming compared multi-view methods in both effectiveness and efficiency.
🏗️Model
<div align="center">
  <img src="model.png" />
</div>

🖥️ Computing Environment
The computing environment includes Ubuntu 20.04.6 LTS, equipped with two AMD EPYC 7763 processors providing 128 physical cores and 256 logical threads, approximately 503 GiB of system memory, and eight NVIDIA GeForce RTX 4090 GPUs with approximately 24 GB of VRAM each.

🧪 Experimental Setup
PyTorch 2.6.0+cu118, Python 3.12.7, CUDA 11.8.

🎞️Experiment
In this experiment, we aim to verify whether the proposed method can effectively purify the data and alleviate the FEB problem.
1 Data
1.1 Multi-view Datasets
We used five multi-view datasets in this experiment:
Datasets	Dataset URL	Password
MVoxCeleb	link	ls12
YoutubeFace	link	ls34
NUS-WIDE-128 (NUS)	link	ls56
Reuters5	link	ls78
Reuters3	link	ls10

For these five multi-view datasets, we only provide the original clean data. You can generate noisy versions with our official noise injection code, or add noise flexibly based on your own experimental needs.

1.2 CIFAR-10N and CIFAR-100N Datasets
The CIFAR-10N and CIFAR-100N datasets used in this work can be downloaded from http://noisylabels.com.
2 Experiment Workflow
To alleviate the FEB problem caused by label noise, we adopt a two-stage strategy of gradient space detection + feature space calibration to purify the given dataset.The specific procedure is as follows:
Step 1: Train the anchor model with the noisy dataset.
Step 2: Perform outlier detection on the noisy dataset and regard outlier samples as potential noisy samples.
Step 3: Remove potential noisy samples from the noisy dataset to obtain a trusted dataset, and retrain the anchor model.
Step 4: Use the retrained anchor model to correct the labels of potential noisy samples.
Step 5: Retrain the anchor model with the purified dataset.
2.1 Gradient and Feature Dual-Perception Framework
2.1.1 Multi-view Datasets
Taking the first fold of Reuters5 as an example, since we manually inject noise into the multi-view dataset, two data files are required during the data preparation phase. One is R5/test_1/noisy dataset, which stores the dataset with injected noise; the other is R5/test_1/purified dataset, which saves the dataset processed by the data purification strategy. This two datasets differ from the original multi-view dataset only in the training labels.
```bash
- python /code-tensorflow/A_Step_0.py
- python /code-tensorflow/A_Step_1.py
- python /code-tensorflow/A_Step_2.py
- python /code-tensorflow/A_Step_3.py
- python /code-tensorflow/A_Step_4.py
- python /code-tensorflow/A_Step_5.py
```
2.1.2 CIFAR Datasets with Label Noise
```bash
- python /code-pytorch/cifar-10-100n/z_step_1.py --noise_type aggre --noise_path /mnt/disk1/lishuai/EA-Dataset/CIFAR-10/CIFAR-10_human.pt --dataset cifar10 --seed 0 --is_human
- python /code-pytorch/cifar-10-100n/z_step_2.py --noise_type aggre --noise_path /mnt/disk1/lishuai/EA-Dataset/CIFAR-10/CIFAR-10_human.pt --dataset cifar10 --seed 0 --is_human
- python /code-pytorch/cifar-10-100n/z_step_3.py --noise_type aggre --noise_path /mnt/disk1/lishuai/EA-Dataset/CIFAR-10/CIFAR-10_human.pt --dataset cifar10 --seed 0 --is_human
- python /code-pytorch/cifar-10-100n/z_step_4.py --noise_type aggre --noise_path /mnt/disk1/lishuai/EA-Dataset/CIFAR-10/CIFAR-10_human.pt --dataset cifar10 --seed 0 --is_human
- python /code-pytorch/cifar-10-100n/z_step_5.py --noise_type aggre --noise_path /mnt/disk1/lishuai/EA-Dataset/CIFAR-10/CIFAR-10_human.pt --dataset cifar10 --seed 0 --is_human
```
2.2 The EMVC Method Based on Purified Dataset
```bash
- python /code-tensorflow/train_tree.py
```

📑Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{li2026Gradient,
title={Evolutionary Multi-View Classification with Label Noise via Gradient and Feature Dual-Perception},
author={Shuai Li, Xinyan Liang, Yuhua Qian, Li Lv},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
}
```


🔬 Related Work
We list below the works most relevant to this paper, including but not limited to the following:<br>
Our research group [link]
Evolutionary deep fusion method and its application in chemical structure recognition, IEEE TEVC21, [paper]
GFD-EMVC: Evolutionary Multi-View Classification with Label Noise via Gradient and Feature Dual-Perception, ICML26, [paper]
Improving evolutionary multi-view classification via eliminating individual fitness bias, NeurIPS25, [paper]
Trusted multi-view classification via evolutionary multi-view fusion, ICLR25, [paper]
DC-NAS: Divide-and-conquer neural architecture search for multi-modal classification, AAAI24, [paper]
Core-structures-guided multi-modal classification neural architecture search, IJCAI24, [paper]
CoMO-NAS: Core-structures-guided multi-objective neural architecture search for multi-modal classification, ACM MM24, [paper]
A fast neural architecture search method for multi-modal classification via knowledge sharing, IJCAI25, [paper]
Multi-scale features are effective for multi-modal classification: An architecture search viewpoint, IEEE TCSVT25, [paper]
Core structure-guided multi-modal classification via monte carlo tree search, IJMLC25, [paper]
LogicNAS: Multi-view neural architecture search method for image sequence logic prediction, IEEE TETCI25, [paper]

Others
Enhancing multimodal learning via hierarchical fusion architecture search with inconsistency mitigation, IEEE TIP25, [paper]
Enhanced protein secondary structure prediction through multi-view multi-feature evolutionary deep fusion method, IEEE TETCI25, [paper]
MFAS: Multimodal fusion architecture search, CVPR19, [paper]
BM-NAS: Bilevel multimodal neural architecture search, AAAI22, [paper]
Deep multimodal neural architecture search, ACMMM20, [paper]
Hierarchical multi-modal fusion architecture search for stock market forecasting, Applied Soft Computing25, [paper]
Harmonic-NAS: Hardware-aware multimodal neural architecture search on resource-constrained devices, ACML23, [paper]
Multi-view information fusion based on federated multi-objective neural architecture search for MRI semantic segmentation, IF25, [paper]
Automatic fused multimodal deep learning for plant identification, Frontiers in Plant Science25, [paper]
<!-- ## 🙏 Acknowledgement -->


📬Contact
If you have any detailed questions or suggestions, you can email us: lishuai_liuzhaona@163.com
