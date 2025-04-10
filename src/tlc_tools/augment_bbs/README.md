# augment_bb_table

Before running this script, make sure you have installed the `tlc_tools` package. Refer to the [README.md](../../README.md) for how do this.

A tool for training bounding box classifiers and extending tables with embeddings and image metrics.

## Usage

```bash
# Train model and process tables
3lc-tools run augment-bb-table --train_table s3://path/to/train --val_table s3://path/to/val

# Train model only
3lc-tools run augment-bb-table --train_table s3://path/to/train --val_table s3://path/to/val --train_only

# Process existing tables with trained model
3lc-tools run augment-bb-table --input_tables "s3://path/to/table1,s3://path/to/table2"

# Only add embeddings (disable metrics)
3lc-tools run augment-bb-table --input_tables "s3://path/to/table" --disable_metrics

# Only add metrics (disable embeddings)
3lc-tools run augment-bb-table --input_tables "s3://path/to/table" --disable_embeddings

# Control memory usage and embedding dimensions
3lc-tools run augment-bb-table --input_tables "s3://path/to/table" --max_memory_gb 16 --reduce_last_dims 1
```

### Arguments

General arguments:

- `--model_name`: Model architecture name (default: "efficientnet_b0")
- `--model_checkpoint`: Path to save/load model checkpoint (default: "./models/bb_classifier.pth")
- `--transient_data_path`: Path for temporary files (default: "./")

Training arguments:

- `--train_table`: Training table URL
- `--val_table`: Validation table URL
- `--train_only`: Only train the model, don't add metrics
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training and embedding collection (default: 32)

Table extension arguments:

- `--input_tables`: Comma-separated list of tables to extend with metrics
- `--disable_embeddings`: Disable adding embeddings to tables
- `--disable_metrics`: Disable adding image metrics to tables
- `--output_suffix`: Suffix for output table names (default: "_extended")

Embedding reduction arguments:

- `--num_components`: Number of PaCMAP components (default: 3)
- `--n_neighbors`: Number of neighbors to consider in PaCMAP (default: 10)
- `--max_memory_gb`: Maximum memory in GB to use for embeddings processing (default: 8)
- `--reduce_last_dims`: Number of dimensions to reduce from the end of embeddings (default: 0)

### Notes

- If `--train_table` and `--val_table` are provided, the script will train a model first
- Model checkpoint is required when adding embeddings without training (default: "./models/bb_classifier.pth")
- If you run out of memory during embedding processing, try reducing the `--max_memory_gb` parameter
- Use `--reduce_last_dims` to trim dimensions from the end of embeddings if needed, takes the mean of the last dimensions
