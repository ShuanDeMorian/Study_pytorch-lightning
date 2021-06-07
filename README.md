# Pytorch Lightning + Hydra

# Pytorch Lightning
## 1. Pytorch Lightning이란 무엇이고 왜 쓰는가?
Pytorch Lightning이란 Pytorch에서 Research에 집중할 수 있도록 나머지 부분(Engineering, Non-essential) 신경을 덜 쓰게 해주는 tool이다.   
  
Pytorch Lightning은 기존의 Pytorch 코드를 Research / Engineering / Non-essential 3가지로 구분하여   
모델 정의 및 학습에 관련된 Research 코드 작성 외의 GPU 설정, 로깅, 실험 설정 등은 기본적으로 제공하여 적은 수정으로 사용할 수 있도록 제공한다.
* Keep all the flexibility (this is all pure PyTorch), but removes a ton of boilerplate
* More readable by decoupling the research code from the engineering
* Easier to reproduce
* Less error-prone by automating most of the training loop and tricky engineering
* Scalable to any hardware without changing your model

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
출처 : https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
1. Use workers in DataLoaders
    ```python
    DataLoader(dataset, num_workers=8)
    ```
    데이터를 로드해 올 때 프로세스를 동시에 여러개 돌려라   
    추천 값은 GPU 개수에 4를 곱한 값이다.   
    num_workers = 4 * num_GPU

2. Pin memory
    ```python
    DataLoader(dataset, pin_memory=True)
    ```
    모델을 돌릴 때 GPU 메모리가 가득찼다고 나오지만 실제 모델이 사용하는 양은 그보다 작을 것이라고 확신이 들 때가 있다.   
    이 overhead를 pinned memory라고 한다. 즉, 메모리 할당량을 고정시켜놓는 것이다.(예약해놓는 것이다.)   
    이 pin_memory를 활성화했을 때의 장점은 CPU가 GPU로 데이터를 전송하는 속도가 빨라진다. 

    ![pin_memory](https://miro.medium.com/max/1050/1*xw4-jfQXEXpLNZFmgMxOQg.png)
    또한 이것은 불필요한 호출을 하지 말아야 한다는 것을 의미한다.
    ```python
    torch.cuda.empty_cache()
    ```
3. Avoid CPU to GPU transfers or vice-versa
    ```python
    # bad
    
    .cpu()
    .item()
    .numpy()
    ```
    위와 같은 방법은 GPU에서 CPU로 data를 옮기는데 이것은 퍼포먼스에 아주 악영향을 끼친다.    
    만약 당신이 computational graph를 clear하고 싶다면 위 방법 대신 .detach()를 사용하라

    ```python
    # good
    
    .detach()
    ```
4. Construct tensors directly on GPU
    대부분의 사람들은 GPU에 tensor를 만들 때 다음과 같이 만든다.
    ```python
    t = tensor.rand(2,2).cuda()
    ```
    그러나 이 방법은 CPU tensor를 만들고 GPU로 옮기는 것이다. 이것은 매우 느리다.   
    대신에 원하는 device에 바로 tensor를 만들어라
    ```python
    t = tensor.rand(2,2, device=torch.device('cuda:0'))
    ```
    만약 당신이 Lightning을 쓴다면 자동으로 알맞은 GPU에 넣어주겠지만   
    코드에서 새로운 tensor를 만든다면 직접 지정해주어야 한다.
    ```python
    t = tensor.rand(2,2, device=self.device)
    ```
    모든 LightningModule은 self.device를 가지고 있는데 이것은 당신이 CPU, multiple GPUs, or TPUs   
    뭘 사용하든 올바른 device를 알려줄 것이다.  

5. Use DistributedDataParallel not DataParallel
    Pytorch는 multiple GPU로 training할 때 두 가지 방법이 있다. 
    * DataParallel (DP)  
        batch를 multiple GPU로 나눈다. 그러나 이것은 model이 각 GPU에 복사되어야 하고   
        gradients가 GPU 0에서 계산된 뒤 다른 GPU들과 sync되어야 한다.   
        이것은 많은 양의 GPU 이동을 발생시키고 오래 걸린다.
    * DistributedDataParallel (DDP)   
        각 GPU에 siloed copy of model을 생성한다(in its own process)   
        그리고 각 GPU에서 가능한 양의 데이터만을 만든다. 이것은 N개의 독립된 model training과 같이 행동하는데   
        다른 점은 각 GPU에서 gradients를 계산한 뒤 모든 모델의 gradients가 sync된다.   
        이것은 각 batch에 한번만 GPU간 data를 전송하는 것이다.

    Lightning에선 두 방법 중 어느 걸 사용할 지 아주 쉽게 결정할 수 있는데 다음과 같다.
    ```python
    Trainer (distributed_backend='ddp', gpus=8)
    Trainer (distributed_backend='dp', gpus=8)
    ```

    DP는 multi-threading, DDP는 multi-processing이다.   
    DP는 하나의 process를 여러 개의 thread가 처리,   
    DDP는 각각 하나의 process를 하나의 thread가 처리한다.   
    
    그렇다면 왜 multi-processing인 DDP 방식이 multi-threading인 DP 방식보다 빠를까?   
    그 이유는 python의 GIL(Global Interpreter Lock) 때문이다. 여러 개의 thread가 동시에 실행되지 못하도록 막는 기능으로 이 때문에 python 언어를 베이스로 하는 pytorch에서 DP가 아닌 DDP를 추천하는 것이다.

6. Use 16-bit precision
    많은 사람들이 사용하진 않지만 Training을 빠르게 할 수 있는 방법이다.   
    32-bit에서 16-bit로 변경하면 다음과 같은 이점이 있다.

    1. Half Memory   
        기존 batch size를 2배로 그러니까 training 시간을 반으로 줄일 수 있다.

    2. 특정 GPU(V100, 2080Ti)에선 더 속도가 빨라진다(3 ~ 8배)   
        왜냐면 이 GPU들은 16-bit 연산에 최적화되어 있기 때문이다.

    Lightning에선 다음과 같이 작동할 수 있다.
    ```python
    Trainer (precision=16)
    ```
    참고로 Pytorch 1.6 이전에서는 Nvidia Apex를 설치해야 한다. 그 이후 버전에서는 16-bit가 default다.   
    Lightning에서는 pytorch 버전을 읽고 자동으로 해준다.

7. Profile your code
    마지막 팁은 Lightning 없이 하기 힘들지도 모른다. 없으면 cprofiler와 같은 것으로 할 수 있다.   
    Lightning에서는 training 중에 한 call 요약본을 얻을 수 있는데 두 가지 방법이 있다.
    1. built-in basic profiler
        ```python
        Trainer(profile=True)
        ``` 
        output은 다음처럼 나온다.
        ![basic_profiler](https://miro.medium.com/max/1050/1*vurXh5DD2-J9WG8A-Cg7cA.png)
    
    2. advanced profiler
        ```python
        proiler = AdvancedProfiler()
        trainer = Trainer (profiler=profiler)
        ```
        ![advanced_profiler](https://miro.medium.com/max/1050/1*OmaLSwe9YevGfYvhmHrM4Q.png)


## 출처
* https://www.pytorchlightning.ai/#top-nav
* https://github.com/PyTorchLightning/pytorch-lightning
* https://www.cognex.com/ko-kr/blogs/deep-learning/research/pytorch-boilerplate
* https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
* https://bbdata.tistory.com/9
   
<br/><br/> 
<br/><br/>
<br/><br/>

# Hydra
![Hydra](https://hydra.cc/img/logo.svg)

## 1. Hydra는 무엇이고 왜 쓰는가?
Hydra란 오픈소스 파이썬 프레임워크로서 연구개발을 간소화시켜준다. Hydra란 이름은 히드라 괴물(머리가 여러개)처럼 비슷한 다수의 작업을 수행할 수 있는 능력에서 비롯되었다.            
주요 기능으로는 가능한 argument의 조합별로 계층적 configuration을 동적(dynamically)으로 생성하고 config file이나 command line를 통해 이를 재정의(override)하는 것이다.      
Hydra를 사용하면 실험에서 쓰이는 argument의 조합을 하나의 명령으로 알아서 해주기 떄문에 실험하기 편리하다.

### Key features
* Hierarchical configuration composable from multiple sources
* Configuration can be specified or overridden from the command line
* Dynamic command line tab completion
* Run your application locally or launch it to run remotely
* Run multiple jobs with different arguments with a single command

## 2. Install
```
pip install hydra-core --upgrade --pre
```

## 3. Combination
여러 개의 설정을 조합하여 테스트하고 싶을 때 아래와 같은 명령어를 사용하면 된다.
hydra가 자동으로 각 조합(3 * 3 = 9개)의 실험을 할 것이다.
```
python lightning_hydra.py -m hidden_size=32,64,128 dropout_rate=0.1,0.3,0.5
```

## 출처
* https://github.com/facebookresearch/hydra
* https://github.com/ashleve/lightning-hydra-template