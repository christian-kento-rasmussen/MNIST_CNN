import torch
from MNIST_CNN import MyAwesomeModel
from MNIST_CNN.data.make_dataset import get_train_loaders

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)

def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model_checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(model_checkpoint)

    _, test_set = get_train_loaders()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in test_set:
            images = images.view(images.shape[0], -1)
            logits = model(images)
            _, predictions = torch.max(logits, dim=1)
            correct += torch.sum(predictions == targets)
            total += targets.shape[0]

    print(f"Accuracy: {correct/total}")

