# Blinder: End-to-end Privacy Protection in Sensing Systems via Personalized Federated Learning
[![arXiv](https://img.shields.io/badge/arXiv-2209.12046-b31b1b.svg)](https://arxiv.org/abs/2209.12046)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/sustainable-computing/Blinder/blob/main/LICENSE.md)

This repository contains the implementation of the paper entitled "Blinder: End-to-end Privacy Protection in Sensing Systems via Personalized Federated Learning".

## Directories:
- [Blinder-Python](https://github.com/sustainable-computing/Blinder/tree/main/Blinder-Python): Source code of Blinder based on PyTorch
- [Blinder-Android](https://github.com/sustainable-computing/Blinder/tree/main/Blinder-Android): Source code of Android deployment, open in Android Studio

## Datasets
Blinder is evaluated on two Human Activity Recognition (HAR) datasets: MotionSense and MobiAct. 

The datasets and the preprocessing script (required for MobiAct) are available at:

- MobiAct V2.0: [Dataset](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2)
    - Preprocessing script: [dataset_builder.py](https://github.com/sustainable-computing/ObscureNet/blob/master/Dataset%26Models/MobiAct%20Dataset/dataset_builder.py)

- MotionSense: [Dataset](https://github.com/mmalekzadeh/motion-sense/tree/master/data)

Note: Pre-trained evaluation models can be found under [Blinder-Python/eval_models/](https://github.com/sustainable-computing/Blinder/tree/main/Blinder-Python/eval_models).



## Dependencies
| Package           | Version       |
| ----------------- |:-------------:| 
| Python3           | 3.8.13        |
| PyTorch           | 1.10.2        |
| TensorFlow        | 2.8.0         |
| imbalanced_learn  | 0.9.0         |
| scikit-learn      | 1.1.2         |


## Installation
### Android App:
Blinder is deployed on Android platforms for real-time data anonymization. This deployment uses Blinder models pre-trained on MobiAct and MotionSense.

- Android Studio project: https://github.com/sustainable-computing/Blinder/tree/main/Blinder-Android

- Android apk file: https://github.com/sustainable-computing/Blinder/releases



## Acknowledgement
- [learn2learn](https://github.com/learnables/learn2learn): a software library for meta-learning research.

## Citation
Xin Yang and Omid Ardakanian. 2023. [Blinder: End-to-end Privacy Protection in Sensing Systems via Personalized Federated Learning](https://doi.org/10.1145/3623397). ACM Trans. Sen. Netw. 20, 1, Article 15 (January 2024), 32 pages.

```
@article{yang2023blinder,
  title={Blinder: End-to-end Privacy Protection in Sensing Systems via Personalized Federated Learning},
  author={Yang, Xin and Ardakanian, Omid},
  journal={ACM Transactions on Sensor Networks},
  volume={20},
  number={1},
  pages={1--32},
  year={2023},
  publisher={ACM New York, NY}
}
```
