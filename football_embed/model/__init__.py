"""Model definitions: event transformer, text branch, metadata MLP, fusion layer.

Architecture overview (4-branch hybrid):
- Event branch: GPT-2 style decoder, 4 layers, 128-dim, 4 heads
- Text branch: ModernBERT-Embed (nomic-ai/modernbert-embed-base), 768d -> 256d
- Metadata branch: 2-layer MLP, output 64-dim
- Fusion: concatenate all branches -> learned projection to 256-384 dim
"""
