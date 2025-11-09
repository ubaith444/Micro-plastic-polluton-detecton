"""Model Evaluation Script"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data import UnderwaterPlasticDataset, get_validation_augmentation, collate_fn
from src.inference import PlasticDetector
from src.training.metrics import DetectionMetrics
from src.utils import setup_logger


def main(args):
    logger = setup_logger('evaluation')
    
    logger.info(f"Loading test dataset from {args.data_dir}...")
    test_dataset = UnderwaterPlasticDataset(
        args.data_dir,
        split='test',
        transforms=get_validation_augmentation(416),
        image_size=416
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    logger.info(f"Loading model from {args.model_path}...")
    detector = PlasticDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    logger.info("Running evaluation...")
    
    print(f"\n{'='*70}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default='data/upd/UPD.v1.yolov5pytorch')
    parser.add_argument('--model_type', default='faster_rcnn')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    main(args)
