from __future__ import annotations

import timm
import tlc
import torch
import torchvision

from PIL.Image import Image

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def transform(sample: tuple[Image, int]) -> tuple[torch.Tensor, int]:
    image = sample[0]
    label = sample[1]

    return (image_transform(image), label)

def main():
    device = "cuda:0"

    # Use a resnet18 model from timm, already trained on CIFAR-10
    model = timm.create_model("hf_hub:FredMell/resnet18-cifar10", pretrained=True).to(device)

    # Load the tables from the tutorial ../../1-create-tables/create-table-from-torch.ipynb
    train_table = tlc.Table.from_names("initial", "CIFAR-10-train", "3LC Tutorials")
    eval_table = tlc.Table.from_names("initial", "CIFAR-10-val", "3LC Tutorials")

    # Apply the transforms to the tables
    train_table = train_table.map(transform)
    eval_table = eval_table.map(transform)

    # Create a 3LC run and collect metrics
    tlc.init(project_name=train_table.project_name, description="Only collect metrics with trained model on CIFAR-10")

    dataloader_args = {"batch_size": 128, "num_workers": 4, "persistent_workers": True, "pin_memory": True}
    classes = list(train_table.get_simple_value_map("Label").values())

    tlc.collect_metrics(
        table=train_table,
        predictor=model,
        metrics_collectors=tlc.ClassificationMetricsCollector(classes=classes),
        dataloader_args=dataloader_args,
        split="train",
    )

    tlc.collect_metrics(
        table=eval_table,
        predictor=model,
        metrics_collectors=tlc.ClassificationMetricsCollector(classes=classes),
        dataloader_args=dataloader_args,
        split="eval",
    )

    tlc.close()

if __name__ == "__main__":
    main()
