from __future__ import annotations

import argparse
import os

import tlc

from tlc_tools.augment_bbs.extend_table_with_metrics import extend_table_with_metrics
from tlc_tools.augment_bbs.finetune_on_crops import train_model
from tlc_tools.cli import register_tool


def parse_table_list(table_string):
    """Parse comma-separated table list"""
    if table_string:
        return [url.strip() for url in table_string.split(",")]
    return None


@register_tool(description="Augment tables with bounding box embeddings and image metrics")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """
    Main function to process tables

    :param args: List of arguments. If None, will parse from command line.
    :param prog: Program name. If None, will use the tool name.
    """
    parser = argparse.ArgumentParser(prog=prog, description="Extend tables with embeddings and image metrics")

    # General arguments
    parser.add_argument("--model_name", default="efficientnet_b0", help="Model architecture name")
    parser.add_argument(
        "--model_checkpoint", default="./models/bb_classifier.pth", help="Path to save/load model checkpoint"
    )
    parser.add_argument("--transient_data_path", default="./", help="Path for temporary files")

    # Training arguments
    parser.add_argument("--train_table", help="Training table URL")
    parser.add_argument("--val_table", help="Validation table URL")
    parser.add_argument("--train_only", action="store_true", help="Only train the model, don't add metrics")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument(
        "--include_background",
        action="store_true",
        default=False,
        help="Whether to train with creating background Bounding boxes",
    )

    # Table extension arguments
    parser.add_argument(
        "--input_tables", type=parse_table_list, help="Comma-separated list of tables to extend with metrics"
    )
    parser.add_argument("--disable_embeddings", action="store_true", help="Disable adding embeddings to tables")
    parser.add_argument("--disable_metrics", action="store_true", help="Disable adding image metrics to tables")
    parser.add_argument("--output_suffix", default="_extended", help="Suffix for output table names")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and embedding collection")
    parser.add_argument("--num_components", type=int, default=3, help="Number of PaCMAP components")
    parser.add_argument("--n_neighbors", type=int, default=10, help="Number of neighbors to consider in PaCMAP")
    parser.add_argument(
        "--max_memory_gb", type=int, default=8, help="Maximum memory in GB to use for embeddings processing"
    )
    parser.add_argument(
        "--reduce_last_dims",
        type=int,
        default=0,
        help="Number of dimensions to reduce from the end of embeddings (0 means no reduction)",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    args = parser.parse_args(tool_args)

    # Check if we're in training mode
    training_mode = args.train_table is not None and args.val_table is not None

    # Validate disable flags
    if args.disable_embeddings and args.disable_metrics and not args.train_only:
        raise ValueError("Cannot disable both embeddings and metrics unless --train_only is specified")

    # Check if model checkpoint exists when not training but adding embeddings
    if not training_mode and not args.disable_embeddings:
        if not os.path.exists(args.model_checkpoint):
            raise ValueError(
                f"Model checkpoint not found at {args.model_checkpoint}. Cannot add embeddings without existing model."
            )
    else:  # In training mode, ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(args.model_checkpoint)
        if checkpoint_dir:  # Only create directory if path has a directory component
            os.makedirs(checkpoint_dir, exist_ok=True)

    # Create transient data directory if it doesn't exist
    os.makedirs(args.transient_data_path, exist_ok=True)

    # Training phase if needed
    if training_mode:
        print("=== Training Model ===")
        print("\nTraining parameters:")
        print(f"  Model: {args.model_name}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Include background: {args.include_background}")
        print(f"  Number of workers: {args.num_workers}")
        print(f"  Model checkpoint: {args.model_checkpoint}")
        print(f"  Train table: {args.train_table}")
        print(f"  Val table: {args.val_table}\n")

        _, best_checkpoint_path = train_model(
            train_table_url=args.train_table,
            val_table_url=args.val_table,
            model_name=args.model_name,
            model_checkpoint=args.model_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            include_background=args.include_background,
            num_workers=args.num_workers,
        )

        if args.train_only:
            print("Training complete. Exiting as --train_only was specified.")
            return

        # Update model_checkpoint to use the best checkpoint path
        args.model_checkpoint = best_checkpoint_path

    # If no input_tables specified, use train and val tables if available
    tables_to_process = args.input_tables
    if not tables_to_process and training_mode:
        tables_to_process = [args.train_table, args.val_table]
        print(f"No input tables specified, using train and val tables: {tables_to_process}")
    elif not tables_to_process:
        raise ValueError("Either --input_tables or both --train_table and --val_table must be specified")

    # Process each table
    pacmap_reducer = None  # Will be created for first table and reused for others
    fit_embeddings = None  # Store embeddings used to fit the reducer

    for i, table_url in enumerate(tables_to_process):
        print(f"\n=== Processing table {i + 1}/{len(tables_to_process)}: {table_url} ===")

        # Load input table
        input_table = tlc.Table.from_url(table_url)
        output_name = f"{input_table.name}{args.output_suffix}"

        output_table_url, pacmap_reducer, fit_embeddings = extend_table_with_metrics(
            input_table=input_table,
            output_table_name=output_name,
            add_embeddings=not args.disable_embeddings,  # Enable unless disabled
            add_image_metrics=not args.disable_metrics,  # Enable unless disabled
            model_checkpoint=args.model_checkpoint if not args.disable_embeddings else None,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_components=args.num_components,
            pacmap_reducer=pacmap_reducer,
            fit_embeddings=fit_embeddings,
            n_neighbors=args.n_neighbors,
            reduce_last_dims=args.reduce_last_dims,
            max_memory_gb=args.max_memory_gb,
        )

        print(f"Created extended table: {output_table_url}")


if __name__ == "__main__":
    main()
