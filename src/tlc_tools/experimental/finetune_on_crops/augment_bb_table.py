from __future__ import annotations

import argparse
import os

import tlc

from tlc_tools.cli.registry import register_tool


def parse_table_list(table_string):
    """Parse comma-separated table list"""
    if table_string:
        return [url.strip() for url in table_string.split(",")]
    return None


@register_tool(experimental=True, description="train a model and extend tables with embeddings and image metrics")
def cli_main(args=None, prog=None):
    """Train a model and extend tables with embeddings and image metrics"""

    from tlc_tools.experimental.finetune_on_crops.extend_table_with_metrics import extend_table_with_metrics
    from tlc_tools.experimental.finetune_on_crops.finetune_on_crops import train_model

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

    args = parser.parse_args(args)

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
        train_model(
            train_table_url=args.train_table,
            val_table_url=args.val_table,
            model_name=args.model_name,
            model_checkpoint=args.model_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        if args.train_only:
            print("Training complete. Exiting as --train_only was specified.")
            return

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
    cli_main()
