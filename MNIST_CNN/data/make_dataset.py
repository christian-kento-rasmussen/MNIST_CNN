#%%
import glob
import torch
from torchvision import transforms

def process_data():
    """Return train and test dataloaders for MNIST."""
    data_folder = "/Users/christian/Documents/School/DTU Human Centered Artificial Intelligence/Machine Learning Operations/MNIST_CNN/data/raw/corruptmnist"
    train, test = [], []
    
    # loads all files in the folder with start train_images_*.pt
    train_images_files = glob.glob(data_folder + "/train_images_*.pt")
    train_targets_files = glob.glob(data_folder + "/train_target_*.pt")
    test_images_files = [data_folder + "/test_images.pt"]
    test_target_files = [data_folder + "/test_target.pt"]

    # loads into arrays
    for img, target in zip(train_images_files, train_targets_files):
        train.append((torch.load(img), torch.load(target)))

    for img, target in zip(test_images_files, test_target_files):
        test.append((torch.load(img), torch.load(target)))

    # concat train list into tensor
    train_images = torch.cat([i[0] for i in train], dim=0)
    train_target = torch.cat([i[1] for i in train], dim=0)
    test_images = torch.cat([i[0] for i in test], dim=0)
    test_target = torch.cat([i[1] for i in test], dim=0)

    # normalize data mean 0 std 1
    normalize_transform = transforms.Normalize(0, 1)
    train_images = normalize_transform(train_images)
    test_images = normalize_transform(test_images)

    return train_images, train_target, test_images, test_target

def get_train_loaders():
    """Return train and test dataloaders for MNIST."""

    # load data from files
    train_images = torch.load("data/processed/corruptmnist/train_images.pt")
    train_target = torch.load("data/processed/corruptmnist/train_target.pt")
    test_images = torch.load("data/processed/corruptmnist/test_images.pt")
    test_target = torch.load("data/processed/corruptmnist/test_target.pt")
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_images, train_target),
        batch_size=64,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_images, test_target),
        batch_size=64,
        shuffle=False
    )
    return train_loader, test_loader

if __name__ == '__main__':
    # Get the data and process it
    print("Processing data...")
    train_images, train_target, test_images, test_target = process_data()

    # saves data to data/processed/
    print("Saving data...")
    torch.save(train_images, "data/processed/corruptmnist/train_images.pt")
    torch.save(train_target, "data/processed/corruptmnist/train_target.pt")
    torch.save(test_images, "data/processed/corruptmnist/test_images.pt")
    torch.save(test_target, "data/processed/corruptmnist/test_target.pt")
    print("Done!")
    pass
# %%
