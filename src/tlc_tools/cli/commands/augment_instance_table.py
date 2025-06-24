from __future__ import annotations

import argparse
import logging
import os

import tlc
from tlc_tools.augment_bbs.extend_table_with_metrics import extend_table_with_metrics
from tlc_tools.augment_bbs.finetune_on_crops import train_model
from tlc_tools.augment_bbs.instance_config import InstanceConfig
from tlc_tools.cli import register_tool
from tlc_tools.cli.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_table_list(table_string):
    """Parse comma-separated table list"""
    if table_string:
        return [url.strip() for url in table_string.split(",")]
    return None


@register_tool(description="Deprecated: Use `3lc-tools augment-instance-table` instead", name="augment_bb_table")
def augment_bb_table(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    raise DeprecationWarning("This tool is deprecated. Use the `3lc-tools augment-instance-table` command instead.")


@register_tool(description="Augment tables with per-instance embeddings and image metrics")
def main(tool_args: list[str] | None = None, prog: str | None = None) -> None:
    """
    Main function to process tables

    This tool performs two main functions:
    1. Train a classifier on instances (bounding boxes or segmentations) from training/validation tables
    2. Extend tables with (optionally) embeddings (dimensionality reduced), predicted labels, and instance-level "
    "image metrics"

    The tool supports a "label-free" mode (--allow_label_free) which uses a pretrained ImageNet model
    for embeddings without requiring training data or labels. This mode cannot train but allows table
    enrichment using the pretrained model.

    :param args: List of arguments. If None, will parse from command line.
    :param prog: Program name. If None, will use the tool name.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Extend tables with per-instance embeddings and image metrics. "
        "This tool can train a classifier on instances (bounding boxes or segmentations) "
        "and/or extend tables with embeddings, predicted labels, and instance-level metrics. "
        "Supports label-free mode using pretrained ImageNet models for embedding extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model and process tables
  3lc-tools run augment-instance-table --train_table s3://path/to/train --val_table s3://path/to/val

  # Train model only
  3lc-tools run augment-instance-table --train_table s3://path/to/train --val_table s3://path/to/val --train_only

  # Process existing tables with trained model
  3lc-tools run augment-instance-table --input_tables "s3://path/to/table1,s3://path/to/table2"

  # Use label-free mode with pretrained model (no training required)
  3lc-tools run augment-instance-table --input_tables "s3://path/to/table" --allow_label_free

  # Only add embeddings (disable metrics)
  3lc-tools run augment-instance-table --input_tables "s3://path/to/table" --disable_metrics

  # Only add metrics (disable embeddings)
  3lc-tools run augment-instance-table --input_tables "s3://path/to/table" --disable_embeddings
        """,
    )

    # General arguments
    parser.add_argument(
        "--model_name",
        default="efficientnet_b0",
        help="Model architecture name (must be available in timm library, default: efficientnet_b0)",
    )
    parser.add_argument(
        "--model_checkpoint",
        default="./models/instance_classifier.pth",
        help="Path to save/load model checkpoint. Not needed for --allow_label_free mode as it uses pretrained model",
    )

    # Training arguments
    parser.add_argument("--train_table", help="Training table URL containing labeled instances for model training")
    parser.add_argument(
        "--val_table", help="Validation table URL containing labeled instances for model validation during training"
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only train the model, don't add metrics to tables. Useful for training models separately from table "
        "processing",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs for the classifier (default: 20)"
    )
    parser.add_argument(
        "--include_background",
        action="store_true",
        default=False,
        help="Whether to train with creating background instances (bounding boxes only). "
        "Creates additional instances from background regions for more robust training",
    )

    # Table extension arguments
    parser.add_argument(
        "--input_tables",
        type=parse_table_list,
        help="Comma-separated list of tables to extend with metrics. "
        "If not specified and training mode is active, will use train and val tables",
    )
    parser.add_argument(
        "--disable_embeddings",
        action="store_true",
        help="Disable adding embeddings to tables. Use when only image metrics are needed",
    )
    parser.add_argument(
        "--disable_metrics",
        action="store_true",
        help="Disable adding image metrics to tables. Use when only embeddings are needed",
    )
    parser.add_argument(
        "--output_suffix",
        default="_extended",
        help="Suffix for output table names (default: _extended). "
        "Output tables will be named as {original_name}{suffix}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and embedding collection (default: 32). " "Reduce if running out of memory",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=3,
        help="Number of PaCMAP components for dimensionality reduction of embeddings (default: 3). "
        "Higher values preserve more information but increase output size",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=10,
        help="Number of neighbors to consider in PaCMAP dimensionality reduction (default: 10). "
        "Affects the quality of the reduced embeddings",
    )
    parser.add_argument(
        "--max_memory_gb",
        type=int,
        default=8,
        help="Maximum memory in GB to use for embeddings processing (default: 8). "
        "Reduce if running out of memory during processing",
    )
    parser.add_argument(
        "--reduce_last_dims",
        type=int,
        default=2,
        help="Number of dimensions to reduce from the end of embeddings (default: 2). "
        "Takes the mean of the last N dimensions. Useful for reducing embedding size "
        "when models output high-dimensional features",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4). "
        "Increase for faster data loading, decrease if experiencing memory issues",
    )

    # Instance handling arguments
    parser.add_argument(
        "--instance_column",
        help="Name of the column containing instances (auto-detect if not specified). "
        "The tool will automatically detect the instance column if not provided",
    )
    parser.add_argument(
        "--instance_type",
        choices=["bounding_boxes", "segmentations", "auto"],
        default="auto",
        help="Type of instances to process (default: auto). "
        "Auto-detect by default, or specify 'bounding_boxes' or 'segmentations' explicitly",
    )
    parser.add_argument(
        "--allow_label_free",
        action="store_true",
        help="Use pretrained ImageNet model for embeddings without requiring training or labels. "
        "Cannot be used for training (requires --train_table and --val_table to be None). "
        "Useful for quick table enrichment with pretrained features",
    )
    # Verbosity control for all commands
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (use -v for debug output, -vv for more verbose)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output except for warnings and errors",
    )
    args = parser.parse_args(tool_args)
    setup_logging(verbosity=args.verbose, quiet=args.quiet)
    logger.debug("Debug logging enabled")

    # Check if we're in training mode
    training_mode = args.train_table is not None and args.val_table is not None

    # Validate disable flags
    if args.disable_embeddings and args.disable_metrics and not args.train_only:
        raise ValueError("Cannot disable both embeddings and metrics unless --train_only is specified")

    # Check for label-free training attempt
    if training_mode and args.allow_label_free:
        logger.info("Warning: Training mode detected with --allow_label_free flag.")
        logger.info("Training requires labels and cannot be performed in label-free mode.")
        logger.info("You can either:")
        logger.info("  1. Remove --allow_label_free to require labels for training")
        logger.info("  2. Use --train_only=false to skip training and only process tables")
        if args.train_only:
            raise ValueError("Cannot train model in label-free mode. Training requires labels.")
        else:
            logger.info("Proceeding with table processing only (skipping training)...")
            training_mode = False

    # Check if model checkpoint exists when not training but adding embeddings (unless label-free mode)
    if not training_mode and not args.disable_embeddings and not args.allow_label_free:
        if not os.path.exists(args.model_checkpoint):
            raise ValueError(
                f"Model checkpoint not found at {args.model_checkpoint}. Cannot add embeddings without existing model. "
                f"Use --allow_label_free to use pretrained model instead."
            )
    else:  # In training mode, ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(args.model_checkpoint)
        if checkpoint_dir:  # Only create directory if path has a directory component
            os.makedirs(checkpoint_dir, exist_ok=True)

    # Training phase if needed
    if training_mode:
        logger.info("=== Training Model ===")

        # Resolve instance configuration for training table
        train_table = tlc.Table.from_url(args.train_table)
        training_instance_config = InstanceConfig.resolve(
            input_table=train_table,
            instance_column=args.instance_column,
            instance_type=args.instance_type,
            allow_label_free=False,  # Training always requires labels
        )

        logger.info("\nTraining parameters:")
        logger.info(f"  Model: {args.model_name}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Include background: {args.include_background}")
        logger.info(f"  Number of workers: {args.num_workers}")
        logger.info(f"  Model checkpoint: {args.model_checkpoint}")
        logger.info(f"  Train table: {args.train_table}")
        logger.info(f"  Val table: {args.val_table}")
        logger.info(f"  Instance column: {training_instance_config.instance_column}")
        logger.info(f"  Instance type: {training_instance_config.instance_type}")
        logger.info(f"  Label column path: {training_instance_config.label_column_path}\n")

        _, best_checkpoint_path = train_model(
            train_table_url=args.train_table,
            val_table_url=args.val_table,
            model_name=args.model_name,
            model_checkpoint=args.model_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            include_background=args.include_background,
            num_workers=args.num_workers,
            instance_config=training_instance_config,
        )

        if args.train_only:
            logger.info("Training complete. Exiting as --train_only was specified.")
            return

        # Update model_checkpoint to use the best checkpoint path
        args.model_checkpoint = best_checkpoint_path

    # If no input_tables specified, use train and val tables if available
    tables_to_process = args.input_tables
    if not tables_to_process and training_mode:
        tables_to_process = [args.train_table, args.val_table]
        logger.info(f"No input tables specified, using train and val tables: {tables_to_process}")
    elif not tables_to_process:
        raise ValueError("Either --input_tables or both --train_table and --val_table must be specified")

    # Process each table
    pacmap_reducer = None  # Will be created for first table and reused for others
    fit_embeddings = None  # Store embeddings used to fit the reducer

    for i, table_url in enumerate(tables_to_process):
        logger.info(f"\n=== Processing table {i + 1}/{len(tables_to_process)}: {table_url} ===")

        # Load input table
        input_table = tlc.Table.from_url(table_url)
        output_name = f"{input_table.name}{args.output_suffix}"

        # Resolve instance configuration for this table
        logger.info("Resolving instance configuration...")
        try:
            instance_config = InstanceConfig.resolve(
                input_table=input_table,
                instance_column=args.instance_column,
                instance_type=args.instance_type,
                allow_label_free=args.allow_label_free,
            )
            logger.info(f"  Instance column: {instance_config.instance_column}")
            logger.info(f"  Instance type: {instance_config.instance_type}")
            logger.info(f"  Label column path: {instance_config.label_column_path}")
            logger.info(f"  Label-free mode: {instance_config.allow_label_free}")
        except ValueError as e:
            logger.info(f"Error: Failed to resolve instance configuration: {e}")
            continue

        # Determine model checkpoint: None for label-free (pretrained), actual path otherwise
        model_checkpoint_to_use = None
        if not args.disable_embeddings:
            if args.allow_label_free:
                model_checkpoint_to_use = None  # Use pretrained model
                logger.info(f"  Using pretrained {args.model_name} model (label-free mode)")
            else:
                model_checkpoint_to_use = args.model_checkpoint  # Use trained model
                logger.info(f"  Using trained model from: {args.model_checkpoint}")

        output_table_url, pacmap_reducer, fit_embeddings = extend_table_with_metrics(
            input_table=input_table,
            output_table_name=output_name,
            add_embeddings=not args.disable_embeddings,
            add_image_metrics=not args.disable_metrics,
            model_checkpoint=model_checkpoint_to_use,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_components=args.num_components,
            pacmap_reducer=pacmap_reducer,
            fit_embeddings=fit_embeddings,
            n_neighbors=args.n_neighbors,
            reduce_last_dims=args.reduce_last_dims,
            max_memory_gb=args.max_memory_gb,
            instance_config=instance_config,
        )

        logger.info(f"Created extended table: {output_table_url}")


if __name__ == "__main__":
    main()
