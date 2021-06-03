# Pytorch Lightning
## 1. Pytorch Lightning이란 무엇이고 왜 쓰는가?
Pytorch Lightning이란 Pytorch에서 Research에 집중할 수 있도록 나머지 부분(Engineering, Non-essential) 신경을 덜 쓰게 해주는 도구이다.   
  
Pytorch Lightning은 기존의 Pytorch 코드를 Research / Engineering / Non-essential 3가지로 구분하여   
모델 정의 및 학습에 관련된 Research 코드 작성 외의 GPU 설정, 로깅, 실험 설정 등은 기본적으로 제공하여 적은 수정으로 사용할 수 있도록 제공한다.

Lightning은 다음의 구조를 따라서 reusable, shareable하다.
* Research code (the LightningModule).
* Engineering code (you delete, and is handled by the Trainer).
* Non-essential research code (logging, etc... this goes in Callbacks).
* Data (use PyTorch DataLoaders or organize them into a LightningDataModule).

## 2. Install
```
pip install pytorch-lightning
```

## 7 Tips To Maximize PyTorch Performance


## 출처
* https://www.pytorchlightning.ai/#top-nav
* https://github.com/PyTorchLightning/pytorch-lightning
* https://www.cognex.com/ko-kr/blogs/deep-learning/research/pytorch-boilerplate
* https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
* https://bbdata.tistory.com/9