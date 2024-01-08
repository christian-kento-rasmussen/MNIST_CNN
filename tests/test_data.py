from MNIST_CNN.data.make_dataset import process_data
from tests import _PATH_DATA


def test_process_data():
    print(_PATH_DATA)

    train_images, train_target, test_images, test_target = process_data()

    # Check that none of the outputs are None
    assert train_images is not None, "train_images should not be None"
    assert train_target is not None, "train_target should not be None"
    assert test_images is not None, "test_images should not be None"
    assert test_target is not None, "test_target should not be None"

    # Add more assertions here based on what you expect from your data
