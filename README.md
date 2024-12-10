# Dual Attention Guidance Network for Self-Supervised Monocular Depth Estimation
This repository represents the official implementation of the paper titled " Dual Attention Guidance Network for Self-Supervised Monocular Depth Estimation".

:rocket:The code will be released soon !

## Inference
```python
python3 inference.py \
                    --model-path checkpoints/epoch_25/model.pth \
                    --inference-resize-height 192 \
                    --inference-resize-width 640 \
                    --image-path imgs \
                    --output-path output-img --output-format .png --semantic-guidance --feature-fusion 
```
## Depth Evaluation

```python
python3 eval_depth.py\
        --sys-best-effort-determinism  \
        --model-name "eval_kitti_depth" \
        --model-load checkpoints/epoch_40 \
        --semantic-guidance --feature-fusion --depth-validation-loaders "kitti_zhou_test" 
```

## Segmentation Evaluation

```python
python3 eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-name "eval_kitti" \
        --model-load /checkpoints/epoch_35 \
        --segmentation-validation-loaders "kitti_2015_train" \
        --segmentation-validation-resize-width 640 \
        --segmentation-validation-resize-height 192
        --eval-num-images 1
```

