"""Interactive Demo Script"""

import argparse
import cv2
from src.inference import PlasticDetector
from src.utils import setup_logger


def main(args):
    logger = setup_logger('demo')
    
    logger.info(f"Loading model from {args.model_path}...")
    detector = PlasticDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    logger.info(f"Processing {args.image_path}...")
    detections, vis_image = detector.predict_image(args.image_path, return_visualization=True)
    
    logger.info(f"Found {len(detections)} plastic objects")
    
    for i, det in enumerate(detections, 1):
        logger.info(f"  {i}. {det['class_name']}: {det['confidence']:.2f}")
    
    if args.output_path:
        cv2.imwrite(args.output_path, vis_image)
        logger.info(f"Result saved to {args.output_path}")
    
    if args.display:
        cv2.imshow('Underwater Plastic Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_type', default='faster_rcnn')
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--output_path', default='output.jpg')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    main(args)
