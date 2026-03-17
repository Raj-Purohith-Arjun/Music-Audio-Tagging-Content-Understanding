from src.models.cnn import CNNModel
from src.models.crnn import CRNNModel


def build_model(config: dict):
    """Instantiate model from config dict."""
    arch = config.get("architecture", "cnn")
    kwargs = {
        "num_classes": config.get("num_classes", 8),
        "embedding_dim": config.get("embedding_dim", 128),
        "dropout": config.get("dropout", 0.3),
    }

    if arch == "cnn":
        return CNNModel(**kwargs)
    elif arch == "crnn":
        return CRNNModel(
            hidden_size=config.get("hidden_size", 256),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
