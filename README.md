# zero_shot_reg
Zero-shot learning for Referring Expression Generation


# data

- referring expressions: refcoco_refdf.json.gz
 (see names_in_context/data)
- bounding boxes: mscoco_bbdf.json.gz
  (see names_in_context/data)
- training-test splits: see data/refcoco_splits.json
- visual features: mscoco_vgg19.npz
  (dropbox)

# code

1) run preprocessing: prepare_refcoco.py
2) run model: src/experiment_refcoco.py adapted from here:
https://arxiv.org/abs/1708.02043

