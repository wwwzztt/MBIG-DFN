# MBIG-DFN
The code for 'Motion Blur Information Guided Dual-Domain Fusion Network for High-Quality Image Deblurring'.

## Quick Run

## Training
run
 ```
python -m torch.distributed.run --nproc_per_node=4 --master_port=6581 basicsr/train_all.py -opt options/train/MBIG_DFN_48.yml --launcher pytorch
```

## Evaluation
run 
```
python -m torch.distributed.run --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/baseline-test.yml --launcher pytorch
```
