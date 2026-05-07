import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from training.data_module import SequenceDataModule
from training.clt_module import CLTLightningModule
import numpy as np
import random
import torch
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser()
    # Path params (defaults handled in main.sh usually, but keeping safe defaults here)
    parser.add_argument("--data-dir", type=str, required=True, help="Path to .a2m or .parquet file")
    parser.add_argument("--model", type=str, default="Profluent-Bio/progen3-112m", help="Name of ProGen3 model")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for checkpoints/logs")
    
    # Model params
    parser.add_argument("--num-layers", type=int, default=12, help="Total layers in pLM")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-hidden", type=int, default=5120, help="Latent dim per layer")
    
    # Training params
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--auxk", type=int, default=64)
    parser.add_argument("--dead-steps-threshold", type=int, default=10000)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="ProGen3-CLT")
    
    args = parser.parse_args()
    
    # Create output directory
    run_name = f"ProGen3_CLT_L{args.num_layers}_D{args.d_hidden}_K{args.k}"
    run_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=os.path.join(run_output_dir, "wandb")
    )

    ckpt_path = None
    last_ckpt_path = os.path.join(run_output_dir, "checkpoints", "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"Found existing checkpoint at {last_ckpt_path}. Resuming...")
        ckpt_path = last_ckpt_path
    
    # Model
    model = CLTLightningModule(args)
    
    # Data
    data_module = SequenceDataModule(args.data_dir, args.batch_size, num_workers=4)
    
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_output_dir, "checkpoints"),
        filename="clt-{step}-{val/loss:.2f}",
        save_top_k=2,
        monitor="val/loss", 
        mode="min",
        save_last=True
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_devices,
        precision="bf16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        val_check_interval=2500, 
        limit_val_batches=10,
        log_every_n_steps=1,
        strategy="ddp"
    )
    
    trainer.validate(model, data_module, ckpt_path=ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()