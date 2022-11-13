import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from inference_helper import InferenceHelper
from PIL import Image


def infer_depth(model_path: str, input_path: str, output_path: str, img_size: tuple = (960, 576)):
    inferHelper = InferenceHelper(pretrained_path=model_path)
    test_img_names = sorted([str(p) for p in Path(input_path).glob('*')])
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for tin in test_img_names[:1]:
        print("Processing", tin)
        img = Image.open(tin)
        _, pred = inferHelper.predict_pil(img, visualized=False, img_size=img_size)
        plt.imshow(pred.squeeze(), cmap='magma_r')
        plt.savefig(str(output_path / tin.split('/')[-1]))


def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for AdaBins')
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--input_path', type=str, help='Path to input images', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save predictions', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    infer_depth(args.model_path, args.input_path, args.output_path)
