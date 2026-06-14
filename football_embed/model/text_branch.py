#!/usr/bin/env python3
"""ModernBERT-Embed wrapper for the text embedding branch.

Base model: nomic-ai/modernbert-embed-base (149M params, 768d, Apache-2.0).
Projects to 256d via a two-layer projection head (768 -> 512 -> 256).
Supports Matryoshka dimension reduction: truncate to [256, 128, 64].

Freezing strategy: all ModernBERT layers frozen except the last N encoder
layers (default 2). The projection head is always trainable.

Encodes player bios, match reports, and scouting descriptions into dense
vectors aligned with the event and metadata branches via contrastive training.

Usage:
    from football_embed.model.text_branch import TextBranch

    model = TextBranch(device="cpu")
    print(f"Trainable params: {model.get_trainable_params():,}")

    # Single forward pass
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
    encoded = tokenizer(
        ["Creative midfielder with excellent vision"],
        return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    emb = model(encoded["input_ids"], encoded["attention_mask"])  # (1, 256)

    # Batch encode (handles tokenization internally)
    embeddings = model.encode([
        "Creative midfielder with excellent vision",
        "Tall centre-back, dominant in the air",
    ])  # (2, 256) CPU tensor

    # Matryoshka: just slice the output
    emb_128 = embeddings[:, :128]
    emb_64 = embeddings[:, :64]

    # Save and reload
    model.save("checkpoints/text_branch")
    model2 = TextBranch.load("checkpoints/text_branch", device="cpu")
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MATRYOSHKA_DIMS = [256, 128, 64]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextBranch(nn.Module):
    """Wrapper around ModernBERT-Embed for text embedding.

    Loads a pre-trained ModernBERT encoder, freezes most layers, and adds
    a trainable projection head that maps 768-dim hidden states to a
    256-dim L2-normalized embedding. Supports Matryoshka truncation.
    """

    # Prefix prepended to NL search queries at inference time (NV-Embed style).
    # Document texts (player descriptions) are encoded without prefix.
    QUERY_PREFIX = "search_query: "

    def __init__(
        self,
        model_name: str = "nomic-ai/modernbert-embed-base",
        output_dim: int = 256,
        projection_hidden: int = 512,
        freeze_except_last_n: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.projection_hidden = projection_hidden
        self.freeze_except_last_n = freeze_except_last_n
        self._device_str = device

        # Load pre-trained encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        encoder_dim = self.encoder.config.hidden_size  # 768 for modernbert-embed-base

        # Projection head: Linear -> GELU -> Linear
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, output_dim),
        )

        # Freeze encoder layers
        self._apply_freezing()

        self.to(device)

    def _apply_freezing(self):
        """Freeze all encoder parameters except the last N layers."""
        # Freeze everything first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last N encoder layers
        # ModernBertModel stores layers directly at self.layers (not nested under .encoder)
        layers = self.encoder.layers
        n_layers = len(layers)
        unfreeze_from = max(0, n_layers - self.freeze_except_last_n)

        for layer in layers[unfreeze_from:]:
            for param in layer.parameters():
                param.requires_grad = True

    def _mean_pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool over non-padding tokens.

        Args:
            last_hidden_state: (B, T, D) encoder output.
            attention_mask: (B, T) with 1 for real tokens, 0 for padding.

        Returns:
            (B, D) pooled representation.
        """
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (last_hidden_state * mask).sum(dim=1)  # (B, D)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return summed / counts

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns L2-normalized embeddings of shape (batch, output_dim).

        Args:
            input_ids: (B, T) token IDs from the tokenizer.
            attention_mask: (B, T) mask (1 = real token, 0 = padding).

        Returns:
            (B, output_dim) L2-normalized embeddings.
        """
        encoder_out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        pooled = self._mean_pool(encoder_out.last_hidden_state, attention_mask)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode a list of texts into embeddings. Returns (N, output_dim) CPU tensor.

        Handles tokenization, batching, and device transfer internally.

        Args:
            texts: list of strings to encode.
            batch_size: number of texts per forward pass.

        Returns:
            (N, output_dim) float32 tensor on CPU.
        """
        self.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = encoded["input_ids"].to(self._device_str)
            attention_mask = encoded["attention_mask"].to(self._device_str)

            emb = self.forward(input_ids, attention_mask)  # (batch, output_dim)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_queries(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode NL search queries with asymmetric prefix (NV-Embed style).

        Prepends QUERY_PREFIX to each text before encoding. Use this for
        natural language retrieval queries. Document/player descriptions
        should use encode() without prefix.

        Args:
            texts: list of NL query strings.
            batch_size: number of texts per forward pass.

        Returns:
            (N, output_dim) float32 tensor on CPU.
        """
        prefixed = [self.QUERY_PREFIX + t for t in texts]
        return self.encode(prefixed, batch_size=batch_size)

    def get_trainable_params(self) -> int:
        """Count parameters that require grad."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------

    def save(self, path: str | Path):
        """Save projection head state_dict and config to a directory.

        The base encoder weights are not saved (loaded from HuggingFace on
        reload). Only the projection head and unfrozen encoder layer weights
        are saved, keeping checkpoints small.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "model_name": self.model_name,
            "output_dim": self.output_dim,
            "projection_hidden": self.projection_hidden,
            "freeze_except_last_n": self.freeze_except_last_n,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

        # Save trainable weights only (projection head + unfrozen encoder layers)
        trainable_names = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        trainable_state = {
            k: v for k, v in self.state_dict().items() if k in trainable_names
        }
        torch.save(trainable_state, path / "trainable_weights.pt")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> TextBranch:
        """Load a saved TextBranch from a checkpoint directory.

        Reconstructs the model from config, loads the base encoder from
        HuggingFace, then overwrites projection head and unfrozen encoder
        layers from the saved weights.

        Args:
            path: directory containing config.json and trainable_weights.pt.
            device: target device ("cpu", "cuda", "mps").

        Returns:
            TextBranch instance with loaded weights.
        """
        path = Path(path)
        config = json.loads((path / "config.json").read_text())

        model = cls(
            model_name=config["model_name"],
            output_dim=config["output_dim"],
            projection_hidden=config["projection_hidden"],
            freeze_except_last_n=config["freeze_except_last_n"],
            device=device,
        )

        saved_state = torch.load(
            path / "trainable_weights.pt", map_location=device, weights_only=True,
        )
        model.load_state_dict(saved_state, strict=False)

        return model


# ---------------------------------------------------------------------------
# Player Card Projector
# ---------------------------------------------------------------------------

class ZoneClassifier(nn.Module):
    """Auxiliary zone classification head on text embeddings.

    Takes 256-dim L2-normalized text embeddings and predicts zone label.
    Used as multi-task auxiliary loss to force zone separation in embedding space.
    """

    def __init__(self, input_dim: int = 256, num_zones: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.num_zones = num_zones
        self.head = nn.Linear(input_dim, num_zones)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict zone logits from text embeddings. Input: (B, D), output: (B, num_zones)."""
        return self.head(x)


class ArchetypeClassifier(nn.Module):
    """Auxiliary archetype classification head on text embeddings.

    Takes 256-dim L2-normalized text embeddings and predicts archetype label
    (one of ~26 sub-archetypes across 7 zones). Used as multi-task auxiliary
    loss to force within-zone discrimination in embedding space.
    """

    def __init__(self, input_dim: int = 256, num_archetypes: int = 26):
        super().__init__()
        self.input_dim = input_dim
        self.num_archetypes = num_archetypes
        self.head = nn.Linear(input_dim, num_archetypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict archetype logits from text embeddings. Input: (B, D), output: (B, num_archetypes)."""
        return self.head(x)


class SkillClassifier(nn.Module):
    """Auxiliary multi-label skill classification head on text embeddings.

    Takes 256-dim L2-normalized text embeddings and predicts 12 soft skill
    tags (float 0-1). Trained with BCE loss to capture cross-cutting player
    skills (aerial, dribbling, set-piece, etc.) that span multiple zones
    and archetypes.
    """

    def __init__(self, input_dim: int = 256, n_skills: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.n_skills = n_skills
        self.fc = nn.Linear(input_dim, n_skills)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict skill logits from text embeddings. Input: (B, D), output: (B, n_skills)."""
        return self.fc(x)  # returns logits, BCE loss handles sigmoid


class PlayerCardProjector(nn.Module):
    """Projects low-dimensional player-card vectors into the text embedding space.

    Takes an N-dim L2-normalized player stat vector and projects it to
    output_dim, then L2-normalizes. Supports two modes:

    - **MLP** (hidden_dim is not None): card_dim -> hidden_dim -> GELU ->
      Dropout -> output_dim -> L2-norm. Default hidden_dim=512, dropout=0.1.
    - **Linear** (hidden_dim is None): card_dim -> output_dim -> L2-norm.
      Used for backward compatibility with old checkpoints.
    """

    def __init__(
        self,
        card_dim: int = 20,
        output_dim: int = 256,
        hidden_dim: int | None = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.card_dim = card_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if hidden_dim is None:
            self.linear = nn.Linear(card_dim, output_dim)
            self.mlp = None
        else:
            self.linear = None
            self.mlp = nn.Sequential(
                nn.Linear(card_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize. Input: (B, card_dim), output: (B, output_dim)."""
        if self.mlp is not None:
            return F.normalize(self.mlp(x), p=2, dim=-1)
        return F.normalize(self.linear(x), p=2, dim=-1)

    def save(self, path: str | Path):
        """Save projector weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "card_dim": self.card_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }
        (path / "card_projector_config.json").write_text(json.dumps(config, indent=2))
        torch.save(self.state_dict(), path / "card_projector_weights.pt")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PlayerCardProjector":
        """Load projector from saved checkpoint.

        Backward compatible: old checkpoints without hidden_dim/dropout keys
        are loaded as single-linear projectors (hidden_dim=None).
        """
        path = Path(path)
        config = json.loads((path / "card_projector_config.json").read_text())
        proj = cls(
            card_dim=config["card_dim"],
            output_dim=config["output_dim"],
            hidden_dim=config.get("hidden_dim", None),
            dropout=config.get("dropout", 0.1),
        )
        state = torch.load(path / "card_projector_weights.pt", map_location=device, weights_only=True)
        proj.load_state_dict(state)
        proj.to(device)
        return proj
