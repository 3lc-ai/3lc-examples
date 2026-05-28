from __future__ import annotations

from typing import Callable

import tlc
import torch
import torch.utils.data
import tqdm
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel

from tlc_tools import add_columns_to_table
from tlc_tools.common import infer_torch_device


def add_embeddings_to_table(
    table: tlc.Table,
    model: torch.nn.Module | None = None,
    embedding_extraction_fn: Callable | None = None,
    embedding_column: str = "embedding",
    batch_size: int = 4,
    device: str | torch.device | None = None,
    preprocess_fn: Callable | None = None,
    image_column: str = "image",
) -> tlc.Table:
    """
    Adds embeddings to a table using a specified model and optional
    dimensionality reduction.


    :param table: The table containing images.
    :param model: Model to use for embedding extraction. If None, defaults to
        ViT from Hugging Face.
    :param embedding_extraction_fn: Function to extract embeddings from the
        model output. If None, defaults to selecting the [CLS] token for ViT.
    :param embedding_column: Column name to add embeddings to in the table.
    :param batch_size: Batch size for processing images.
    :param device: Device to use for inference.
    :param preprocess_fn: Preprocessing function applied to each sample before
        being passed to the model. If model is None, this argument is ignored
        and the function defaults to extracting ``image_column`` from the
        sample dict and applying a standard ViT preprocessing pipeline.
    :param image_column: Name of the column from which to read images when
        using the default preprocessing pipeline. If not present in the
        sample, falls back to the first available "image"/"Image" key.

    :returns: Table with an added column containing embeddings.
    """
    device = torch.device(device or infer_torch_device())

    # Load default model if none provided
    if model is None:
        model_name = "google/vit-base-patch16-224"
        model = ViTModel.from_pretrained(model_name).to(device)  # type: ignore[arg-type]
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
            ]
        )

        def preprocess_fn(sample):
            if image_column in sample:
                image = sample[image_column]
            elif "image" in sample:
                image = sample["image"]
            elif "Image" in sample:
                image = sample["Image"]
            else:
                msg = f"Sample has no '{image_column}', 'image', or 'Image' key: {list(sample)}"
                raise KeyError(msg)

            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            return image_transform(image)

        embedding_extraction_fn = lambda output: output.last_hidden_state[:, 0, :].cpu().numpy()  # noqa: E731

    if embedding_extraction_fn is None:
        embedding_extraction_fn = lambda output: output.cpu().numpy() if isinstance(output, torch.Tensor) else output  # noqa: E731

    # Build a non-mutating view that applies the preprocessing on read
    view = table.with_transform(preprocess_fn) if preprocess_fn is not None else table

    dataloader = torch.utils.data.DataLoader(view, batch_size=batch_size, shuffle=False)  # type: ignore[arg-type,var-annotated]

    all_embeddings = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Extracting embeddings"):
        with torch.no_grad():
            outputs = model(batch.to(device))
            embeddings = embedding_extraction_fn(outputs)

        all_embeddings.extend(embeddings.tolist())

    # We assign a special number role to the embedding column so that it will be
    # automatically selected for dimensionality reduction, and will not be sent
    # to the Dashboard for visualization.
    embedding_size = len(all_embeddings[0])
    # We don't want the embedding to be displayed in the "sample-view" of the table; not editable post-computation.
    embedding_schema = tlc.schemas.EmbeddingSchema(shape=(embedding_size,), sample_type="hidden", writable=False)

    extended_table = add_columns_to_table(
        table=table,
        columns={embedding_column: all_embeddings},
        schemas={embedding_column: embedding_schema},
        output_table_name="with_embeddings",
    )
    return extended_table
