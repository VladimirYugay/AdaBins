# AdaBins

Implementation of the depth model for Quo Vadis: [**Is Trajectory Forecasting the Key Towards Long-Term Multi-Object Tracking?**](https://arxiv.org/pdf/2210.07681.pdf)

The implementation is based on [AdaBins](https://github.com/shariqfarooq123/AdaBins). The model is trained on a huge pedestrian synthetic dataset (MOTSynth) designed for monocular multi-object tracking. It supports images of a higher resolution, and has a custom data augmentation for simulating mirroring surfaces.

## Download links
* You can download the pretrained model "AdaBins_synthetic.pt" from [here](https://drive.google.com/file/d/1HMQJI01n3ncH8mOxb3-F3uQX0fNsg83h/view?usp=sharing)
* You can download the MOTSynth dataset [here](https://motchallenge.net/data/MOTSynth-MOT-CVPR22/)

## Setting things up

```
conda env create --name adabins --file=environment.yml
conda activate adabins
```

## Inference

In `scripts` there are scripts for inferences on a single image, as well as on `MOTSynth`, `MOTS` datasets. 

To run inference on a folder of images:

```
python scripts/infer.py --model_path <checkpoint_path_pt> --input_path <images_dir> --output_path <output_dir>
```

To run training on MOTSynth dataset:

```
python train.py --data_path <path_to_motsynth> --gt_path <path_to_motsynth_depth> --data_path_eval <path_to_motsynth> --gt_path_eval <path_to_motsynth_depth>  --bs 24 --epochs 50 --input_height 576 --input_width 960 --validate_every 1 --filenames_file ./train_test_inputs/adabins_motsynth/motsynth_train.txt --filenames_file_eval ./train_test_inputs/adabins_motsynth/motsynth_test.txt --same-lr --name adabins_for_motsynth`
```
