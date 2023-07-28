torchrun --nproc_per_node=6 train.py \
  --data-path /workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/ \
  --model resnet152 -b 42 --output-dir runs/`date +%s`

