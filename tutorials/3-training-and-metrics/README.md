# Training and Metrics Collection

This folder contains notebooks demonstrating how to work with Runs and metrics in 3LC.

|  |  |  |
|:----------:|:----------:|:----------:|
| **Collect metrics** | **SAM embeddings** | **Dimensionality reduction** |
| [![train-image-classification](../images/collect-only.jpg)](collect-metrics.ipynb) | [![train-object-detection](../images/sam-embeddings.jpg)](train-object-detection.ipynb) | [![dim-red](../images/dimensionality-reduction.jpg)](dimensionality-reduction.ipynb) |
| This notebook covers using a trained image classification model for metrics collection. Collects the metrics "predicted_label", "confidence", and "loss".| This notebook demonstrates extracting embeddings using the SAM model, and adding those embeddings to a Run. | Reduce dimensionality of embeddings in tables, using a single reduction model to transform several tables. |
| **Autoencoder** | | |
| [![train-image-classification](../images/autoencoder.jpg)](train-autoencoder.ipynb) | |  |
| Train an autoencoder to use for embedding extraction. Collect image reconstructions and embeddings from the trained model. |  |  |
