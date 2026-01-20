# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a randomly initialized YOLOv5 model to ONNX format.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.yolo import Model
from utils.general import LOGGER, check_yaml, print_args
from utils.torch_utils import select_device


def export_onnx(model, im, file, opset=12, dynamic=False, simplify=True):
    """Export a YOLOv5 model to ONNX format with random weights."""
    try:
        import onnx
        from onnx import version_converter
    except ImportError:
        raise ImportError('ONNX required for export. Install with: pip install onnx onnx-simplifier')

    LOGGER.info(f'\nONNX: starting export with onnx {onnx.__version__}...')
    f = str(file.with_suffix('.onnx'))

    # Export
    torch.onnx.export(
        model.cpu() if dynamic else model,
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output0'],
        dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}, 'output0': {0: 'batch', 1: 'anchors'}} if dynamic else None,
    )

    # Check ONNX model
    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)

    # Add metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            import onnxslim
            LOGGER.info(f'ONNX: simplifying with onnxslim {onnxslim.__version__}...')
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'ONNX: simplifier failure: {e}')

    LOGGER.info(f'ONNX: export success ✅, saved as {f}')
    return f, model_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5n.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='image size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    parser.add_argument('--dynamic', action='store_true', help='dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')
    parser.add_argument('--name', default='yolov5n-random', help='model name')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)
    print_args(vars(opt))

    device = select_device(opt.device)

    # Create model with random weights
    LOGGER.info(f'Creating randomly initialized model from {opt.cfg}...')
    model = Model(opt.cfg).to(device)

    # Example input
    im = torch.rand(opt.batch_size, 3, opt.imgsz, opt.imgsz).to(device)

    # Export
    output_path = Path(ROOT) / opt.name
    export_onnx(model, im, output_path, opt.opset, opt.dynamic, opt.simplify)


if __name__ == '__main__':
    main()