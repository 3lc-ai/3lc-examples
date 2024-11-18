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
    embedding_column: str = "embedding",
    batch_size: int = 4,
    device: str | torch.device | None = None,
    preprocess_fn: Callable | None = None,
) -> tlc.Table:
    """
    Adds embeddings to a table using a specified model and optional dimensionality reduction.


    :param table: The table containing images.
    :param model: Model to use for embedding extraction. If None, defaults to ViT from Hugging Face.
    :param embedding_extraction_fn: Function to extract embeddings from the model output. If None, defaults to selecting the [CLS] token for ViT.
    :param embedding_column: Column name to add embeddings to in the table.
    :param batch_size: Batch size for processing images.
    :param device: Device to use for inference.
    :param preprocess_fn: Preprocessing function for images. If None, defaults to a standard ViT preprocessing pipeline.

    :returns: Table with an added column containing embeddings.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load default model if none provided
    if model is None:
        model_name = "google/vit-base-patch16-224"
        model = ViTModel.from_pretrained(model_name).to(device)
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        preprocess_fn = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
            ]
        )
        embedding_extraction_fn = lambda output: output.last_hidden_state[:, 0, :]

    # Map the table to ensure samples are compatible with the model
    table.map(preprocess_fn)

    # Set up DataLoader
    dataloader = torch.utils.data.DataLoader(table, batch_size=batch_size, shuffle=False)

    # Run inference and extract embeddings
    all_embeddings = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Extracting embeddings"):
        with torch.no_grad():
            outputs = model(batch.to(device))
            embeddings = embedding_extraction_fn(outputs).cpu().numpy()

        all_embeddings.extend(embeddings.tolist())

    # We assign a special number role to the embedding column so that it will be
    # automatically selected for dimensionality reduction, and will not be sent
    # to the Dashboard for visualization.
    embedding_size = len(all_embeddings[0])
    embedding_schema = tlc.Schema(
        value=tlc.Float32Value(number_role=tlc.NUMBER_ROLE_NN_EMBEDDING),
        size0=tlc.DimensionNumericValue(embedding_size, embedding_size),
        sample_type="hidden",  # We don't want the embedding to be displayed in the "sample-view" of the table
        writable=False,  # We do not allow editing the embedding values after they have been computed
    )

    extended_table = add_columns_to_table(
        table=table,
        columns={embedding_column: all_embeddings},
        schemas={embedding_column: embedding_schema},
        output_table_name="with_embeddings",
    )
    return extended_table
