"""Batch Prediction Script"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.inference import PlasticDetector
from src.utils import setup_logger


def main(args):
    logger = setup_logger('batch_predict')
    
    logger.info(f"Loading model from {args.model_path}...")
    detector = PlasticDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    input_dir = Path(args.input_dir)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    logger.info(f"Found {len(image_files)} images")
    
    all_detections = {}
    for image_path in tqdm(image_files, desc="Processing"):
        detections = detector.predict_image(image_path)
        all_detections[str(image_path)] = detections
    
    output_path = Path(args.output_dir) / 'predictions.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--model_type', default='faster_rcnn')
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    main(args)
