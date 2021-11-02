# [AI 해커톤 본선 과제-2] 작물 내 해충 객체 검출 

## Requirements
- Ubuntu  : 18.04
- Python  : 3.8.5
```
pip install -r requirements.txt
```
 
## Model
- CBNetV2: A Composite Backbone Network Architecture for Object Detection
- Paper : https://arxiv.org/abs/2107.00420 (Tingting Liang, Xiaojie Chu, Yudong Liu, Yongtao Wang, Zhi Tang, Wei Chu, Jingdong Chen, Haibin Ling)
>CBNetV2는 Assistant backbone과 Lead backbone으로 구분된 동일한 구조의 K개(K≥1)의 Backbone을 Composite Function과 Assistant Supervision을 활용한 Object Detection 모델입니다. 이때 Backbone Network의 구조는 기존에 존재하는 Pre-trained Backbones의 Architecture와 Weight를 활용하여 추가적인 Pre-training 없이 높은 성능을 기록한 모델입니다. 
> 
>Assistant Backbones에서 생성한 High-, Low-Level Feature을 Composite Function을 통해 Lead Backbone의 Receptive Field를 점차 넓혀 성능을 향상시키고자 하였습니다. Composition function으로는 DenseNet의 Dense Connection에서 영감을 받은 Dense Higher-Level Composition(DHLC)을 활용하여 이전 Backbone의 Higher-Level Stage Feature를 다음 Backbone의 Lower-Level Stage Feature에 더해주었습니다.
> 
>CBNetV2는 Assistant Backbone에 대한 Supervision을 활용하는 학습 전략을 통해 기존 CBNet 보다 성능을 향상 시켰습니다. 기존의 Lead backbone의 Feature를 학습한 Detection Head 1의 Loss와 더불어 Assistant Supervision을 학습한 Detection Head 2의 Loss를 함께 계산합니다. 이때, Detection Head 1과 2는 가중치를 공유합니다. Lambda를 통해 Assistant Loss에 대한 가중치를 조절합니다. 
>
>Detection Head는 Faster R-CNN에 Feature Pyramid Network를 적용하여 사용하였으며, 2개의 pre-trained Swin-Large Transformer로 구성된 Dual-Swin-L 모델을 통해 COCO test-dev 데이터에 대하여 59.4% box AP와 51.6% mask AP를 기록하였습니다.

## Preprocessing
```
your code
```

## Training
```
your code
```
## Ensemble
- Weighted Boxes Fusion: ensembling boxes for object detection models
- Paper : https://arxiv.org/abs/1910.13302 (Roman Solovyev, Weimin Wang, Tatiana Gabruseva)
> Weighted Boxes Fusion(WBF)는 여러 Object Detection 모델이 예측한 Bounding Box를 모두 활용하여 Ensemble하는 방법입니다. Interest Over Union(IoU)가 특정 threshold 이상인 bounding box에 대하여 융합을 진행합니다. 융합된 bounding box의 좌표는 각 bounding box의 confidence score의 weighted sum으로 계산하여 confidence score가 높은 box에 더 많이 영향을 받도록합니다..

```
your code
```

