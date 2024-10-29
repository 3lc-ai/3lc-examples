from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import tlc
import torch
import torch.utils.data
import tqdm
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel

from tools import add_columns_to_table


def add_embeddings_to_table(
    table: tlc.Table,
    model: torch.nn.Module | None = None,
    embedding_extraction_fn: Callable | None = None,
    image_column: str = "image",
    embedding_column: str = "embedding",
    batch_size: int = 4,
    device: str | torch.device | None = None,
    preprocess_fn: Callable | None = None,
) -> tlc.Table:
    """
    Adds embeddings to a table using a specified model and optional dimensionality reduction.

    Parameters:
        table (tlc.Table): The table containing images.
        model (torch.nn.Module, optional): Model to use for embedding extraction.
            If None, defaults to ViT from Hugging Face.
        embedding_extraction_fn (Callable, optional): Function to extract embeddings from the model output.
            If None, defaults to selecting the [CLS] token for ViT.
        image_column (str): Column name of images in the table.
        embedding_column (str): Column name to add embeddings to in the table.
        label_column (str, optional): Label column name if needed for embedding stratification.
        batch_size (int): Batch size for processing images.
        device (str or torch.device, optional): Device to use for inference.
        preprocess_fn (Callable, optional): Preprocessing function for images.
            If None, defaults to a standard ViT preprocessing pipeline.

    Returns:
        tlc.Table: Table with an added column containing embeddings.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load default model if none provided
    if model is None:
        model_name = "google/vit-base-patch16-224"
        model = ViTModel.from_pretrained(model_name).to(device)
        embedding_extraction_fn = embedding_extraction_fn or (lambda output: output.last_hidden_state[:, 0, :])

    # Prepare preprocessing function
    if preprocess_fn is None:
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        preprocess_fn = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
            ]
        )

    def map_fn(sample):
        return preprocess_fn(sample[0])

    all_embeddings = []
    # Set up DataLoader
    table.map(map_fn)
    dataloader = torch.utils.data.DataLoader(table, batch_size=batch_size, shuffle=False)

    # Run inference and extract embeddings
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Extracting embeddings"):
        with torch.no_grad():
            outputs = model(batch.to(device))
            embeddings = embedding_extraction_fn(outputs).cpu().numpy()

        all_embeddings.extend(embeddings.tolist())

    extended_table = add_columns_to_table(table, {embedding_column: all_embeddings})
    return extended_table
