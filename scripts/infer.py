from pathlib import Path

import matplotlib.pyplot as plt
from inference_helper import InferenceHelper
from PIL import Image

if __name__ == '__main__':
    path = "checkpoints/vladimir_depth/vlad_depth.pt"
    img_size = (960, 576)
    inferHelper = InferenceHelper(pretrained_path=path)
    test_img_names = sorted([str(p) for p in Path('test_imgs').glob('*')])
    output_path = Path('test_outputs')
    output_path.mkdir(parents=True, exist_ok=True)
    for tin in test_img_names[:1]:
        print("Processing", tin)
        img = Image.open(tin)
        centers, pred = inferHelper.predict_pil(img, visualized=False, img_size=img_size)
        plt.imshow(pred.squeeze(), cmap='magma_r')
        plt.savefig(str(output_path / tin.split('/')[-1]))
