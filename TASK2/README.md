# [AI 해커톤 본선 과제-1] 과수(사과, 배) 화상병 진단 문제

## Requirements
- Python 3.#.#
- Ubuntu 
- Cuda
 
```
pip install -r requirements.txt
```
 
## Model
- CBNetV2: A Composite Backbone Network Architecture for Object Detection
- Paper : https://arxiv.org/abs/2107.00420
> CBNetV2는 pre-training 없이 기존에 존재하는 pre-trained backbones의 architecture와 weight를 최대한 활용한 모델로 assistant backbone과 lead backbone으로 구분된 동일한 구조의 K개(K≥1)의 backbone을 Composite Function과 Assistant Supervision을 활용하여 fine-tuning한 Object Detection 모델입니다. 
Assistant backbones에서 생성한 high-, low-level feature을 composite function을 통해 lead backbone의 receptive field를 점차 넓혀 성능을 향상시키고자 하였습니다. Composition function으로는 DenseNet의 Dense Connection에서 영감을 받은 Dense Higher-Level Composition(DHLC)을 활용하여 이전 backbone의 higher-level stage feature를 다음 backbone의 lower-level stage feature에 더해주었습니다.
CBNetV2는 assistant backbone에 대한 supervision을 활용하여 학습 전략을 통해 기존 CBNet의 성능을 향상 시켰습니다. 기존의 lead backbone의 feature를 학습한 detection head 1뿐만 아니라 assistant supervision을 학습한 detection head 2를 통해 loss를 계산합니다. 이때, Detection head 1과 2는 가중치를 공유합니다. Lambda를 통해 assistant loss에 대한 가중치를 조절합니다. 
Detection Head는 Faster R-CNN에 Feature Pyramid Network를 적용하여 사용하였으며, 2개의 pre-trained Swin-Large Transformer로 구성된 Dual-Swin-L 모델을 통해 COCO test-dev 데이터에 대하여 59.4% box AP와 51.6% mask AP를 기록하였습니다.

## Preprocessing
```
your code
```

## Training
```
your code
```
