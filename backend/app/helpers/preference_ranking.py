"""
Preference ranking module for protein embeddings using Bradley-Terry models.

This module implements preference learning models based on the Bradley-Terry
approach, which learns preferences from pairwise comparisons. The implementation
uses PyTorch to build MLP models that predict preference scores from embeddings.

The key idea is to train a model to predict scalar preference scores for each
protein variant. These scores can then be used to rank variants, with higher
scores indicating more preferred/active variants.

Within each mini-batch during training, we generate all possible pairwise
preferences based on the activity labels. This approach efficiently leverages
the structure in the dataset, as each embedding is only processed once per batch,
but contributes to multiple preference pairs.
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from folde.util import get_top_percentile_recall_score
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for batch-based preference learning.

    This dataset handles embedding samples and their corresponding activity measurements.
    The training loop will compute all valid preferences within each mini-batch.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        activity_labels: np.ndarray,
        device: str | None = None,
    ):
        """Initialize the preference dataset.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings for all samples
            activity_labels: Array of shape (n_samples,) with activity measurements for all samples
        """
        self.embeddings: torch.Tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        self.activity_labels: torch.Tensor = torch.tensor(
            activity_labels, dtype=torch.float32, device=device
        )
        self.n_samples: int = len(embeddings)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple containing (embedding, activity_label)
        """
        return self.embeddings[idx], self.activity_labels[idx]


class BradleyTerryMLP(nn.Module):
    """Bradley-Terry model for preference learning using an MLP.

    This model maps embeddings to scalar preference scores using a multi-layer
    perceptron. The Bradley-Terry model predicts the probability that item i is
    preferred over item j as sigmoid(score_i - score_j).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        activation: nn.Module = nn.ReLU(),
    ):
        """Initialize the Bradley-Terry MLP model.

        Args:
            embedding_dim: Dimension of the input embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function to use
        """
        super().__init__()

        # Build MLP layers
        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final layer to scalar preference score
        layers.append(nn.Linear(prev_dim, 1, bias=False))  # no bias)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute preference score.

        Args:
            x: Tensor of shape (batch_size, embedding_dim) with embeddings

        Returns:
            Tensor of shape (batch_size, 1) with preference scores
        """
        return self.mlp(x)


def batch_bradley_terry_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    pair_mask: torch.Tensor | None,
    importance_sampling_reweighting_strat: str | None,
    importance_sampling_temperature: float | None,
):
    B = scores.size(0)

    # compute pairwise logits & targets
    diff = scores.view(-1, 1) - scores.view(1, -1)  # (B, B)
    Y = (labels.view(-1, 1) > labels.view(1, -1)).float()  # (B, B)

    # get scores and weights
    s = scores.view(-1, 1)  # (B,1)
    s_i = s.expand(B, B).t()  # (B,B)
    s_j = s.expand(B, B)  # (B,B)

    if importance_sampling_reweighting_strat == "min":
        w_full = torch.exp(torch.min(s_i, s_j) / importance_sampling_temperature)
    elif importance_sampling_reweighting_strat == "max":
        w_full = torch.exp(torch.max(s_i, s_j) / importance_sampling_temperature)
    else:
        w_full = torch.ones_like(diff)  # uniform

    if pair_mask is None:
        pair_mask = torch.ones_like(diff, dtype=torch.int).triu(diagonal=1)
    pair_mask_bool = pair_mask.bool()

    # compute weighted BCE for a given mask
    logits = diff[pair_mask_bool]
    targets = Y[pair_mask_bool]
    w = w_full[pair_mask_bool]
    w = w / w.sum()  # normalize within this split
    w = w.detach()
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (w * losses).sum()

    return loss


def get_random_pair_split(
    B: int, labels: np.ndarray, rng_seed: int, val_fraction: float = 0.2, device=None
):
    """
    Split directed pairs into training vs validation such that validation
    masks only include pairs for which there is NO directed chain
    through any intermediate in the training graph. This ensures that
    the validation loss only contains nontrivial comparisons.

    Returns:
      train_mask, val_mask : (BxB) boolean masks
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    # sample an initial undirected training graph with probability 1 - val_fraction
    prob_train = 1 - val_fraction
    train_graph = torch.rand((B, B), device=device) < prob_train
    train_graph.fill_diagonal_(False)

    # build directed adjacency from training_graph + labels for BFS
    labels_np = labels.tolist()
    adj = [[] for _ in range(B)]
    for u in range(B):
        for v in range(B):
            if train_graph[u, v] and labels_np[u] > labels_np[v]:
                adj[u].append(v)

    # do BFS
    reach = [[False] * B for _ in range(B)]
    for u in range(B):
        visited = [False] * B
        stack = [u]
        visited[u] = True
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not visited[y]:
                    visited[y] = True
                    stack.append(y)
        reach[u] = visited

    # collect all non‐trivial directed pairs (i→j) with no path for validation loss
    nontrivial = [(i, j) for i in range(B) for j in range(B) if i != j and not reach[i][j]]

    # sample exactly M = val_fraction * B*(B-1) of them for validation, note that B*(B-1) gets rid of the diagonal
    M = int(val_fraction * B * (B - 1))
    M = min(M, len(nontrivial))
    logging.debug(f"Sampling {M} validation pairs from {len(nontrivial)} non-trivial pairs")
    selected = random.sample(nontrivial, M)

    # build boolean masks as before
    val_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
    for i, j in selected:
        val_mask[i, j] = True
    train_mask = ~val_mask
    train_mask.fill_diagonal_(False)
    val_mask.fill_diagonal_(False)

    return train_mask, val_mask


class PreferenceTrainer:
    """Trainer for Bradley-Terry preference models.

    This class handles the training and evaluation of Bradley-Terry models,
    including data preparation, training loops, and evaluation metrics.
    """

    def __init__(
        self,
        model: BradleyTerryMLP,
        random_state: int,
        device: str | None = None,
    ):
        """Initialize the preference trainer.

        Args:
            model: Bradley-Terry model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training ('cpu' or 'cuda')
                If None, will use CUDA if available, otherwise CPU
        """
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state

        self.model = model.to(self.device)
        self.scaler = GradScaler()  # Add this line

    def train(
        self,
        train_embeddings: np.ndarray,
        train_activity_labels: np.ndarray,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        patience: int | None,
        use_mse_loss: bool,
        importance_sampling_reweighting_strat: str | None,
        importance_sampling_temperature: float | None,
        use_exponential_learning_rate_decay: bool,
        use_plateau_learning_rate_decay: bool,
        val_embeddings: np.ndarray | None = None,
        val_activity_labels: np.ndarray | None = None,
        do_validation_with_pair_fraction: float | None = None,
        val_frequency: int = 10,
        test_embeddings: np.ndarray | None = None,
        test_activity_labels: np.ndarray | None = None,
        batch_test_size: int = 5000,
    ) -> dict[str, Any]:
        """Train the Bradley-Terry model using batch-based training.

        Args:
            train_embeddings: Array of shape (n_samples, embedding_dim) with embeddings for all samples
            train_activity_labels: Array of shape (n_samples,) with activity measurements for all samples
            batch_size: Number of samples to include in each batch
            epochs: Maximum number of epochs to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            patience: Number of epochs to wait for validation improvement before early stopping
            use_mse_loss: Whether to use MSE loss instead of Bradley-Terry loss
            do_importance_sampling: Whether to use importance sampling / reweighting in BT loss
            val_embeddings: embeddings for validation set or None
            val_activity_labels: activity labels for validation set or None
            val_frequency: Frequency of validation runs
        Returns:
            Dictionary with training metrics:
                'train_loss': List of training losses for each epoch
                'val_loss': List of validation losses for each epoch
        """

        torch.manual_seed(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Create datasets and dataloaders
        train_dataset = PreferenceDataset(
            train_embeddings, train_activity_labels, device=self.device
        )
        val_dataset = None
        if val_embeddings is not None and val_activity_labels is not None:
            val_dataset = PreferenceDataset(val_embeddings, val_activity_labels, device=self.device)
        test_dataset = None
        if test_embeddings is not None and test_activity_labels is not None:
            test_dataset = PreferenceDataset(
                test_embeddings, test_activity_labels, device=self.device
            )

        shuffle_train_batches = True
        train_mask, val_mask = None, None
        if do_validation_with_pair_fraction is not None:
            assert val_embeddings is None and val_activity_labels is None
            shuffle_train_batches = False
            if batch_size < train_activity_labels.shape[0]:
                raise ValueError(
                    f"Batch size {batch_size} is less than the number of training samples {train_activity_labels.shape[0]}."
                )
            train_mask, val_mask = get_random_pair_split(
                train_activity_labels.shape[0],
                train_activity_labels,
                self.random_state,
                do_validation_with_pair_fraction,
                device=self.device,
            )

        # Create datasets and dataloaders with fixed random seed
        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train_batches,
            drop_last=False,
            generator=g,
            # Irrelevant for GPU-loaded datasets.
            # persistent_workers=True,
            # num_workers=4,
            # pin_memory=True,
        )

        metrics: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "test_recall_1pct": []}

        best_val_loss = float("inf")
        best_val_loss_epoch = 0
        best_model_state = None

        val_loss, test_recall_1pct = self.evaluate(
            train_dataset,
            val_dataset,
            test_dataset,
            val_mask,
            importance_sampling_reweighting_strat,
            importance_sampling_temperature,
            batch_test_size,
        )
        metrics["train_loss"].append(np.inf)
        metrics["val_loss"].append(val_loss if val_loss is not None else np.nan)
        metrics["test_recall_1pct"].append(
            test_recall_1pct if test_recall_1pct is not None else np.nan
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        exponential_lr_schedule = None
        plateau_lr_schedule = None
        if use_exponential_learning_rate_decay:
            exponential_lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif use_plateau_learning_rate_decay:
            plateau_lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10, verbose=True
            )

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_number, (batch_embeddings, batch_activity_labels) in enumerate(train_loader):
                batch_embeddings = batch_embeddings  # .to(self.device)
                batch_activity_labels = batch_activity_labels  # .to(self.device)

                optimizer.zero_grad()

                # "Autocase" is the first part of automatic mixed precision training, which gives a speedup.
                with autocast():
                    scores = self.model(batch_embeddings)

                    if use_mse_loss:
                        loss = F.mse_loss(scores.squeeze(-1), batch_activity_labels)
                    else:
                        loss = batch_bradley_terry_loss(
                            scores,
                            batch_activity_labels,
                            train_mask,
                            importance_sampling_reweighting_strat,
                            importance_sampling_temperature,
                        )

                    if loss.item() == 0:
                        # logger.warning(f'Zero loss encountered in batch {batch_number} in epoch {epoch} with {len(batch_embeddings)} members.')
                        logger.warning(
                            f"Zero loss encountered in batch {batch_number} in epoch {epoch} with {len(batch_embeddings)} members. "
                            f"Score variance: {scores.var().detach().cpu().item()}; "
                            f"activity level variance: {batch_activity_labels.var().detach().cpu().item()}; "
                            f"train mask occupancy: {'NA' if train_mask is None else (train_mask.sum() / train_mask.numel()).item()}"
                        )

                # "Scaler" is the second part of automatic mixed precision training.
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss += loss.item()
                num_batches += 1

            if exponential_lr_schedule is not None:
                exponential_lr_schedule.step()

            # Average training loss
            if num_batches > 0:
                train_loss /= num_batches

            # Validation - only run every val_frequency epochs
            is_val_round = epoch == 0 or ((epoch + 1) % val_frequency == 0)
            if is_val_round:
                metrics["train_loss"].append(train_loss)

                val_loss, test_recall_1pct = self.evaluate(
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    val_mask,
                    importance_sampling_reweighting_strat,
                    importance_sampling_temperature,
                    batch_test_size,
                )
                metrics["val_loss"].append(val_loss if val_loss is not None else np.nan)
                metrics["test_recall_1pct"].append(
                    test_recall_1pct if test_recall_1pct is not None else np.nan
                )

                if val_loss is not None:
                    if plateau_lr_schedule is not None:
                        plateau_lr_schedule.step(val_loss, epoch=epoch)

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_loss_epoch = epoch
                        best_model_state = {
                            k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                        }

                    if patience is not None and epoch - best_val_loss_epoch >= patience:
                        # logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        if best_model_state is not None:
            with torch.no_grad():
                self.model.load_state_dict(
                    {k: v.to(self.device) for k, v in best_model_state.items()}
                )

        return metrics

    def evaluate(
        self,
        train_dataset: PreferenceDataset,
        val_dataset: PreferenceDataset | None = None,
        test_dataset: PreferenceDataset | None = None,
        val_mask: torch.Tensor | None = None,
        importance_sampling_reweighting_strat: str | None = None,
        importance_sampling_temperature: float | None = None,
        batch_test_size: int = 5000,
    ) -> tuple[float | None, float | None]:
        """Evaluate the model using batch-based evaluation without leaking GPU memory.

        Args:
            train_dataset: Dataset containing only the training samples
            val_and_train_dataset: Dataset containing the union of train + val samples (or None)
            test_dataset: Dataset containing held‑out test samples (or None)

        Returns:
            Tuple ``(val_loss, test_recall_1pct)``. Either element may be ``None`` if the
            corresponding dataset was not supplied.
        """
        self.model.eval()

        # Ensure we never build a computation graph and aggressively free intermediates.
        with torch.no_grad():
            val_loss: float | None = None
            test_recall_1pct: float | None = None

            if val_dataset is not None:
                if val_mask is not None:
                    raise ValueError(
                        "Cannot specify both a validation dataset and a validation mask"
                    )

                train_and_val_embeddings = torch.cat(
                    [train_dataset.embeddings, val_dataset.embeddings], dim=0
                )
                train_and_val_scores = self.model(train_and_val_embeddings)

                # Make a mask that ignores the block of training pairs in the loss calculation.
                ignore_train_loss_mask = torch.ones(
                    (train_and_val_scores.shape[0], train_and_val_scores.shape[0]), dtype=torch.int
                )
                ignore_train_loss_mask[
                    : train_dataset.embeddings.shape[0], : train_dataset.embeddings.shape[0]
                ] = 0
                ignore_train_loss_mask = ignore_train_loss_mask.triu(diagonal=1)

                activity_labels = torch.cat(
                    [train_dataset.activity_labels, val_dataset.activity_labels], dim=0
                )

                # Chunk the evaluation to avoid OOM with large datasets
                total_samples = train_and_val_scores.shape[0]
                chunk_losses = []
                for start_idx in range(0, total_samples, batch_test_size):
                    end_idx = min(start_idx + batch_test_size, total_samples)

                    # Get chunks for this batch
                    scores_chunk = train_and_val_scores[start_idx:end_idx]
                    labels_chunk = activity_labels[start_idx:end_idx]
                    mask_chunk = ignore_train_loss_mask[start_idx:end_idx, start_idx:end_idx]

                    chunk_loss_tensor = batch_bradley_terry_loss(
                        scores_chunk,
                        labels_chunk,
                        mask_chunk,
                        importance_sampling_reweighting_strat=importance_sampling_reweighting_strat,
                        importance_sampling_temperature=importance_sampling_temperature,
                    )
                    chunk_losses.append(float(chunk_loss_tensor.detach().cpu().item()))
                    del chunk_loss_tensor
                    torch.cuda.empty_cache()

                # Average the losses across chunks
                val_loss = float(np.mean(chunk_losses))

                del (
                    train_and_val_embeddings,
                    train_and_val_scores,
                    ignore_train_loss_mask,
                    activity_labels,
                )
                torch.cuda.empty_cache()
            elif val_mask is not None:
                train_scores = self.model(train_dataset.embeddings)
                val_loss_tensor = batch_bradley_terry_loss(
                    train_scores,
                    train_dataset.activity_labels,
                    val_mask,
                    importance_sampling_reweighting_strat=importance_sampling_reweighting_strat,
                    importance_sampling_temperature=importance_sampling_temperature,
                )
                val_loss = float(val_loss_tensor.detach().cpu().item())
                del train_scores, val_loss_tensor
                torch.cuda.empty_cache()

            if test_dataset is not None:
                # Move only the scores to CPU for metric computation; labels are already on CPU.
                test_scores_cpu = self.model(test_dataset.embeddings).detach().cpu()

                test_recall_1pct = get_top_percentile_recall_score(
                    test_dataset.activity_labels.detach().cpu().numpy(),
                    test_scores_cpu.numpy(),
                    1.0,
                )
                del test_scores_cpu
                torch.cuda.empty_cache()

        return val_loss, test_recall_1pct

    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict preference scores for embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim) with embeddings

        Returns:
            Array of shape (n_samples,) with preference scores
        """
        self.model.eval()
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores: np.ndarray = self.model(embeddings_tensor).squeeze(-1).cpu().numpy()

        return scores


def create_preference_model(
    embedding_dim: int,
    hidden_dims: list[int],
    dropout: float = 0.2,
    device: str | None = None,
    random_state: int = 0,
) -> Tuple[BradleyTerryMLP, PreferenceTrainer]:
    """Create a preference model and trainer with standard parameters.

    This is a convenience function to quickly set up a model and trainer
    with reasonable default parameters.

    Args:
        embedding_dim: Dimension of the input embeddings
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to use for training ('cpu' or 'cuda')
            If None, will use CUDA if available, otherwise CPU

    Returns:
        Tuple containing (model, trainer)
    """
    # This... might be where we should set seeds for initialization of model weights.
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    model = BradleyTerryMLP(embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
    # model = torch.compile(model)

    trainer = PreferenceTrainer(
        model=model,
        random_state=random_state,
        device=device,
    )

    return model, trainer
