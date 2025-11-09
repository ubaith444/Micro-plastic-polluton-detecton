"""Training Script for Underwater Plastic Detection"""

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import UnderwaterPlasticDataset, collate_fn
from src.data import get_training_augmentation, get_validation_augmentation
from src.models import FasterRCNNDetector
from src.training import Trainer
from src.utils import setup_logger, create_dirs


def main(args):
    create_dirs(args.output_dir, args.log_dir)
    logger = setup_logger('training', args.log_dir)
    
    logger.info("="*70)
    logger.info("UNDERWATER PLASTIC DETECTION - TRAINING")
    logger.info("="*70)
    logger.info(f"Dataset: Underwater Plastic Dataset (UPD)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Image size: {args.img_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info("="*70 + "\n")
    
    device = torch.device(args.device)
    
    # Load datasets
    logger.info("📦 Loading UPD dataset...")
    
    train_dataset = UnderwaterPlasticDataset(
        root_dir=args.data_dir,
        split='train',
        transforms=get_training_augmentation(args.img_size),
        image_size=args.img_size
    )
    
    val_dataset = UnderwaterPlasticDataset(
        root_dir=args.data_dir,
        split='val',
        transforms=get_validation_augmentation(args.img_size),
        image_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"✓ Train samples: {len(train_dataset)}")
    logger.info(f"✓ Val samples: {len(val_dataset)}\n")
    
    # Display dataset statistics
    train_stats = train_dataset.get_dataset_statistics()
    logger.info(f"📊 Training Set Statistics:")
    logger.info(f"   Total boxes: {train_stats['total_boxes']}")
    logger.info(f"   Avg boxes/image: {train_stats['avg_boxes_per_image']:.2f}")
    logger.info(f"   Plastic: {train_stats['category_distribution']['plastic']}")
    logger.info(f"   Trash: {train_stats['category_distribution']['trash']}\n")
    
    # Model
    logger.info(f"🤖 Loading model: {args.model}...")
    model = FasterRCNNDetector(num_classes=3, pretrained=True)  # 2 classes + background
    model = model.to(device)
    logger.info(f"✓ Model loaded\n")
    
    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        use_tensorboard=True
    )
    
    # Train
    logger.info("🚀 Starting training...")
    logger.info("="*70 + "\n")
    
    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_frequency=args.save_frequency
    )
    
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train underwater plastic detector')
    
    # Dataset
    parser.add_argument('--data_dir', default='data/upd/UPD.v1.yolov5pytorch', help='UPD dataset directory')
    
    # Model
    parser.add_argument('--model', default='faster_rcnn', choices=['faster_rcnn', 'yolov8'])
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    # Output
    parser.add_argument('--output_dir', default='runs/training')
    parser.add_argument('--log_dir', default='logs')
    
    # Hardware
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Checkpointing
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_frequency', type=int, default=10)
    
    args = parser.parse_args()
    main(args)
