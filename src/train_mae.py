import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

# Import PCEN from our utility file
from utils import PCEN

# --- 1. Configuration ---

class MAEConfig:
    DATA_PATH = 'data/preprocessed_spectrograms'
    N_MELS = 128
    MODEL_NAME = 'vit_base_patch16_224'
    IMG_SIZE = (N_MELS, 800)
    PATCH_SIZE = 16
    MASKING_RATIO = 0.75
    BATCH_SIZE = 32
    LEARNING_RATE = 1.5e-4 # Restore to the more effective rate
    EPOCHS = 20
    NUM_WORKERS = 0

# --- 2. PyTorch Dataset ---
class PreprocessedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        return torch.load(file_path)

# --- 3. Lightning Module (The MAE Model) ---
# --- 3. Lightning Module (SIMPLIFIED & STABLE) ---
# Add this at the top of src/train_mae.py
# Add this at the top of src/train_mae.py if it's not already there
import math

# --- 3. Lightning Module (with a more robust scheduler) ---
# --- 3. Lightning Module (FINAL ARCHITECTURE) ---
class LitAudioMAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # --- Encoder ---
        self.encoder = timm.create_model(config.MODEL_NAME, pretrained=False, in_chans=1, img_size=config.IMG_SIZE)
        
        # --- Decoder ---
        # A simple transformer decoder
        decoder_dim = 512 # Can be smaller than encoder's 768
        self.decoder_embed = nn.Linear(self.encoder.embed_dim, decoder_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # This is the crucial missing piece: positional embeddings for the decoder
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)

        # A few transformer blocks for the decoder
        self.decoder_blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(decoder_dim, num_heads=16, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(4) # 4 blocks is a common choice
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, config.PATCH_SIZE**2, bias=True)
        
        # Loss
        self.l1_loss = nn.L1Loss()

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        x = self.encoder.patch_embed(x.unsqueeze(1))
        x = x + self.encoder.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.config.MASKING_RATIO)
        # We don't use the class token in the encoder for MAE
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Add the decoder positional embeddings (the "blueprint")
        x = x_ + self.decoder_pos_embed[:, 1:, :] # Skip cls token pos embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        pred = self.decoder_pred(x)
        return pred

    def forward_loss(self, target_spec, pred_patches, mask):
        p = self.config.PATCH_SIZE
        target_patches = target_spec.unfold(1, p, p).unfold(2, p, p).reshape(target_spec.shape[0], -1, p*p)
        mask = mask.unsqueeze(-1).expand_as(target_patches)
        loss = self.l1_loss(pred_patches[mask.bool()], target_patches[mask.bool()])
        return loss

    def training_step(self, batch, batch_idx):
        target_spec = batch
        latent, mask, ids_restore = self.forward_encoder(target_spec)
        pred_patches = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(target_spec, pred_patches, mask)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.LEARNING_RATE)
        
        try:
            train_loader_len = len(self.trainer.train_dataloader)
        except (AttributeError, TypeError):
            print("Estimating dataloader length for scheduler setup...")
            dataset_size = len(PreprocessedDataset(self.config.DATA_PATH))
            train_loader_len = dataset_size // self.config.BATCH_SIZE
            if dataset_size % self.config.BATCH_SIZE != 0:
                train_loader_len += 1

        num_training_steps = train_loader_len * self.config.EPOCHS
        num_warmup_steps = train_loader_len

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        if avg_loss is not None:
            print(f"\n--- Epoch {self.current_epoch} Summary ---")
            print(f"Average Training Loss: {avg_loss:.4f}")
            print("---------------------------\n")
            
            
            
# --- 4. Main Execution ---
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    config = MAEConfig()
    
    dataset = PreprocessedDataset(config.DATA_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    model = LitAudioMAE(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/mae_checkpoints',
        filename='rainforest-mae-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3, monitor='train_loss', mode='min', save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS, accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback], gradient_clip_val=1.0
    )
    
    print("✅ Starting Rainforest-MAE pre-training...")
    trainer.fit(model, dataloader)
    print("🎉 Rainforest-MAE pre-training finished!")