# AdaBins

Implementation of the depth model for Quo Vadis: [**Is Trajectory Forecasting the Key Towards Long-Term Multi-Object Tracking?**](https://arxiv.org/pdf/2210.07681.pdf)

The implementation is based on [AdaBins](https://github.com/shariqfarooq123/AdaBins). The model is trained on a huge pedestrian synthetic dataset (MOTSynth) designed for monocular multi-object tracking. It supports images of a higher resolution, and has a custom data augmentation for simulating mirroring surfaces.

## Download links
* You can download the pretrained model "AdaBins_synthetic.pt" from [here](https://drive.google.com/file/d/1HMQJI01n3ncH8mOxb3-F3uQX0fNsg83h/view?usp=sharing)
* You can download the MOTSynth dataset [here](https://motchallenge.net/data/MOTSynth-MOT-CVPR22/)

## Setting things up

`conda env create --name adabins --file=environment.yml`

`conda activate adabins`

## Inference

In `scripts` there are scripts for inferences on a single image, as well as on `MOTSynth`, `MOTS` datasets. 

To run inference on a folder of images:

`python scripts/infer.py --model_path checkpoints/vladimir_depth/<model_name>.pt --input_path test_imgs/ --output_path test_output`


