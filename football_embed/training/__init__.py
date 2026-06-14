"""Training scripts, data pair generation, and loss functions.

Training pipeline:
1. Train event transformer on SPADL sequences (next-action prediction)
2. Fine-tune text branch with contrastive loss (optional)
3. Train fusion model end-to-end with contrastive pairs
"""
