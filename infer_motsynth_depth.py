from __future__ import absolute_import, division, print_function

import argparse
import os
from pathlib import Path

import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

from infer import InferenceHelper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--data_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--output_path', type=str,
                        help='Output path', required=True)
    parser.add_argument('--seq_file_path', type=str,
                        help='Path to the file with sequences names')
    parser.add_argument('--model_path', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--img_width', default=640,
                        help='input image width')
    parser.add_argument('--img_height', default=480,
                        help='input image height')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter; see README.md for an example"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inferHelper = InferenceHelper(pretrained_path=args.model_path)
    inferHelper.device = device
    model_name = str(Path(args.model_path).parts[-1])
    output_path = Path(args.output_path)
    data_path = Path(args.data_path)
    with open(args.seq_file_path, 'r') as seq_file:
        seqs = ["{:0>3s}".format(line.strip()) for line in seq_file.readlines()]

    for seq_path in sorted(data_path.glob('*')):
        if seq_path.parts[-1] not in seqs:
            continue
        output_seq_path = output_path / seq_path.parts[-1] / model_name
        output_seq_path.mkdir(parents=True, exist_ok=True)
        print("Processing sequence:", seq_path.parts[-1])
        with torch.no_grad():
            for img_path in sorted(seq_path.glob('*')):
                # Load image and preprocess
                input_image = pil.open(str(img_path)).convert('RGB')
                input_image = input_image.resize((args.img_width, args.img_height), pil.LANCZOS)
                _, depth = inferHelper.predict_pil(input_image, visualized=False)
                depth = depth[0, 0, ...]
                depth_file_name = str(img_path.parts[-1]).split('.')[0]
                np.savez_compressed(str(output_seq_path / depth_file_name), depth)


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
