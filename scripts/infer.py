import argparse
from pathlib import Path
import os
import cv2
import numpy as np 

import matplotlib.pyplot as plt
from inference_helper import InferenceHelper
from PIL import Image
from typing import List


def infer_depth(model_path: str, input_path: str, output_path: str,
                images: List[str], save_as_png: False,  img_size: tuple = (960, 576), ):

    inferHelper = InferenceHelper(pretrained_path=model_path)

    if len(images) > 0:
        test_img_names = [os.path.join(input_path, img) for img in images]
    else:
        test_img_names = sorted([str(p) for p in Path(input_path).glob('*')])
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for tin in test_img_names:
        print("Processing", tin)
        img = Image.open(tin)
        _, pred = inferHelper.predict_pil(
            img, visualized=False, img_size=img_size)

        depth_image = cv2.resize(pred[0, 0], dsize=(
            img.width, img.height), interpolation=cv2.INTER_LANCZOS4)

        image_name = os.path.splitext(tin.split('/')[-1])[0]
        np.save(os.path.join(output_path, f'{image_name}.npy'), depth_image)
        if save_as_png:

            im = Image.fromarray((depth_image/np.max(depth_image)*255).astype(np.uint8)).convert("L")
            im.save(os.path.join(output_path, f'{image_name}.png'))
        

def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script for AdaBins')
    parser.add_argument('--model_path', type=str,
                        help='Path to the model', required=True)
    parser.add_argument('--input_path', type=str,
                        help='Path to input images', required=True)
    parser.add_argument('--output_path', type=str,
                        help='Path to save predictions', required=True)
    parser.add_argument('--images', nargs='+', default=[],
                        help='List of individuals images in <images_dir>', required=False)
    parser.add_argument('--save-as-png', action="store_true", default=False,
        help='Save depth image also as png', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    infer_depth(args.model_path, args.input_path,
                args.output_path, args.images, args.save_as_png)
