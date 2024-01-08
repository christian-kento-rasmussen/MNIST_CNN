import torch

from MNIST_CNN.data.make_dataset import get_train_loaders
from MNIST_CNN.models.model import MyAwesomeModel
from MNIST_CNN.predict_model import predict


def test_predict():
    """Test predict function."""

    # load model from models/model.pt
    model_state_dict = torch.load("models/model.pt")
    model = MyAwesomeModel()
    model.load_state_dict(model_state_dict)

    # load data loader
    train_loader, test_loader = get_train_loaders()

    output = predict(model, test_loader)

    assert output is not None, "output should not be None"
