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
This paper studies a fundamental yet often overlooked premise in evolutionary multi-view classification (EMVC): the impact of label noise on EMVC, such as distorting fitness landscapes shaped by individual fitness values (e.g., test accuracy).
Traditional EMVC assumes training labels are noise-free, yet this often fails in practice.
As a result, label noise introduces harmful supervision during the training phase, resulting in distorted fitness landscapes and the emergence of fitness evaluation bias (FEB). This bias misguides the evolutionary trajectory, causing the search process to stagnate in local optima.
Given that label noise largely stems from the mislabeling of samples near their decision boundaries by human annotators, we thus compared the decision boundaries of human annotators and models, and found discrepancies between the two. Based on this observation, we propose a simple yet effective ``detect-then-calibrate" data purification framework that leverages outlier analysis in the gradient space (i.e., treating outliers as noisy samples) and prototype calibration in the feature space (i.e., utilizing feature prototypes of noise-free samples to correct the labels of noisy samples).
Experimental results demonstrate that this strategy can effectively purify the data and alleviate FEB; furthermore, it can improve the performance of various multi-view learning paradigms in label noise scenarios.
🏗️Model
<div align="center">
  <img src="model.png" />
</div>

🖥️ Computing Environment
The computing environment includes Ubuntu 24.04.2 LTS as the operating system, equipped with an AMD EPYC processor with 160 physical cores (320 logical threads), 566 GB of DDR4 memory, and 8 NVIDIA GeForce RTX 5090 GPUs, each with 32 GB of VRAM.

🧪 Experimental Setup
In our experiments,
For the label noise learning comparison experiments, to ensure consistency with existing methods, we implement all experiments using the PyTorch framework (PyTorch 2.8.0+cu128, Python 3.10.18, CUDA 12.8).
For the multi-view learning comparison experiments, we implement all experiments using TensorFlow framework (TensorFlow 2.10.0 GPU, Python 3.9.23, CUDA 11.2).

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
