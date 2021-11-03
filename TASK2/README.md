# [AI 해커톤 본선 과제-2] 작물 내 해충 객체 검출 

## Requirements
- Ubuntu  : 18.04
- Python  : 3.8.5

### Install git
```
pip install git
```
### Download Pre-trained Weight
```
git clone https://github.com/VDIGPKU/CBNetV2.git
cd CBNetV2
wget https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip
unzip cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip
```
### Install Apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```
pip install -r requirements.txt
```
 
## Preprocessing
- CoCo format annotation file 생성함.
```
python preprocess.py
```

## Train
- 모델 Architecture : CBNetV2
- Detection Head : Cascade RCNN
- Pre-Trained Model : Swin Transformer
```
tools/dist_train.sh 
configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py 1
```

### 방법론 1. 
- CBNetV2: A Composite Backbone Network Architecture for Object Detection
- Paper : https://arxiv.org/abs/2107.00420 (Tingting Liang, Xiaojie Chu, Yudong Liu, Yongtao Wang, Zhi Tang, Wei Chu, Jingdong Chen, Haibin Ling)
>CBNetV2는 Assistant backbone과 Lead backbone으로 구분된 같은 구조의 K개(K≥1)의 Backbone을 Composite Function과 Assistant Supervision을 활용한 Object Detection 모델임. 이때 Backbone Network의 구조는 기존에 존재하는 Pre-trained Backbones의 Architecture와 Weight를 활용하여 추가적인 Pre-training 없이도 높은 성능을 기록한 모델임. 
> 
>Assistant Backbones에서 생성한 High-, Low-Level Feature를 Composite Function을 통해 Lead Backbone의 Receptive Field를 점차 넓혀 성능을 향상하고자 함. Composition function으로는 DenseNet의 Dense Connection에서 영감을 받은 Dense Higher-Level Composition(DHLC)을 활용하여 이전 Backbone의 Higher-Level Stage Feature를 다음 Backbone의 Lower-Level Stage Feature에 더함.
> 
>CBNetV2는 Assistant Backbone에 대한 Supervision을 활용하는 학습 전략을 통해 기존 CBNet 보다 성능이 향상함. 기존의 Lead backbone의 Feature를 학습한 Detection Head 1의 Loss와 더불어 Assistant Supervision을 학습한 Detection Head 2의 Loss를 함께 계산합니다. 이때, Detection Head 1과 2는 가중치를 공유함. Lambda를 통해 Assistant Loss에 대한 가중치를 조절함.

### 방법론 2. 
- Cascade R-CNN: Delving into High Quality Object Detection
- Paper : https://arxiv.org/pdf/1712.00726v1.pdf (Zhaowei Cai, Nuno Vasconcelos)
>Cascade R-CNN은 각 연속된 stage마다 IoU threshold를 증가시키며 detector를 학습시킨 Object Detection 모델임. Object Detection에서 IoU threshold는 region proposals의 positive와 negative를 결정하는 Hyper-parameter로 활용되는데, IoU threshold를 낮게 설정하면 bounding box의 좌표가 부정확해지고, 반대로 IoU threshold를 높게 설정하면 bounding box의 좌표가 정확해지지만, detection 성능을 떨어지는 현상이 발생함. 따라서, 낮은 IoU threshold로 학습한 detector의 output으로 보다 높은 IoU threshold로 설정된 detector를 학습시키며 매 stage를 거치며 더 정확한 proposal을 생성하며 detection을 수행하는 모델임.

### Pre-Trained Model 
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- Paper : https://arxiv.org/pdf/2103.14030v2.pdf (Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo)
>Swin Transformer는 Object Detection이나 Semantic Segmentation과 같은 Dense Prediction Task에서 이미지의 시각적 개체에 대한 Scale 변화가 크고, Pixel Resolution이 커서 Vision Transformer(ViT)가 잘 작동하지 않는다는 점을 보완한 모델임. Swin Transformer는 작은 크기의 Patch에서부터 시작하여 Layer가 깊어질수록 점차 주변 Patch를 병합하여 Hierarchical Representation을 구축함. 또한, Window Multi-head Self-Attention(MSA)과 Shifted Window MSA 방법론을 제안하여 Shifted Window가 이전 Layer의 Windows를 연결하여 Dense Prediction Task에서 큰 성능 개선함.

## Inference
## TTA 설명
- Test Time Augmentation
> Test Time Augmentation(TTA)는 Test 이미지에 대하여 Augmentation을 적용하여 Augmented Image 또한 평가하여 최종 분류 결과를 출력하는 기법임. Augmentation 기법 중 Flip을 적용함.

### Confidence Score가 높은 모델 순서대로 TTA Inference 후, Submission format으로 변경함.
### Iteration 66000, Flip = True
```
python image_demo.py '/DATA/02_bugdetection/images/test/*.jpg' configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py  work_dirs/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco/iter_66000.pth --output work_dirs/ --save_name results_cascade_66000_flip
python submission.py --file_dir work_dirs/results_cascade_66000_flip.pickle --save_name submission_cascade_66000_flip.json
```
### Iteration 70000, Flip = True
```
python image_demo.py '/DATA/02_bugdetection/images/test/*.jpg' configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py  work_dirs/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco/iter_70000.pth --output work_dirs/ --save_name results_cascade_70000_flip
python submission.py --file_dir work_dirs/results_cascade_70000_flip.pickle --save_name submission_cascade_70000_flip.json
```

### Iteration 66000, Flip = False
```
python image_demo.py '/DATA/02_bugdetection/images/test/*.jpg' configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py  work_dirs/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco/iter_70000.pth --output work_dirs/ --save_name results_cascade_66000_mod
python submission.py --file_dir work_dirs/results_cascade_66000_mod.pickle --save_name submission_cascade_66000_mod.json
```
### Iteration 70000, Flip = False
```
python image_demo.py '/DATA/02_bugdetection/images/test/*.jpg' configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py  work_dirs/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco/iter_70000.pth --output work_dirs/ --save_name results_cascade_70000_mod
python submission.py --file_dir work_dirs/results_cascade_70000_mod.pickle --save_name submission_cascade_70000_mod.json
```
### Iteration 68000, Flip = False
```
python image_demo.py '/DATA/02_bugdetection/images/test/*.jpg' configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py  work_dirs/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco/iter_70000.pth --output work_dirs/ --save_name results_cascade_68000_mod
python submission.py --file_dir work_dirs/results_cascade_68000_mod.pickle --save_name submission_cascade_68000_mod.json
```
### Iteration 64000, Flip = False
```
python image_demo.py '/DATA/02_bugdetection/images/test/*.jpg' configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py  work_dirs/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco/iter_70000.pth --output work_dirs/ --save_name results_cascade_64000_mod
python submission.py --file_dir work_dirs/results_cascade_64000_mod.pickle --save_name submission_cascade_64000_mod.json
```

## Ensemble
```
python submission_wbf.py --file_list submission_cascade_66000_flip.json submission_cascade_70000_flip.json submission_cascade_66000_mod.json submission_cascade_70000_mod.json submission_cascade_68000_mod.json submission_cascade_64000_mod.json --save_name submission_cascade_wbf_6670flip_iou05_weights332211.json --weights 3 3 2 2 1 1
```
- Weighted Boxes Fusion: ensembling boxes for object detection models
- Paper : https://arxiv.org/abs/1910.13302 (Roman Solovyev, Weimin Wang, Tatiana Gabruseva)
> Weighted Boxes Fusion(WBF)은 여러 Object Detection 모델이 예측한 Bounding Box를 모두 활용하는 Ensemble 기법임. Interest Over Union(IoU)이 특정 threshold 이상인 bounding box에 대하여 융합을 진행함. 융합된 bounding box의 좌표는 각 bounding box의 confidence score의 weighted sum으로 계산하여 confidence score가 높은 box에 더 많이 영향을 받도록 함.

