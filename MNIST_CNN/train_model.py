import click
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from MNIST_CNN import MyAwesomeModel, get_train_loaders


def train(lr, e, checkpoint):
    """Train a model on MNIST."""
    print("Training day and night")
    print("lr, ", lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_dataloader, _ = get_train_loaders()

    discrim = torch.nn.NLLLoss()  # torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    for epoch in tqdm(range(e)):
        epoch_loss = 0
        for images, targets in train_dataloader:
            # add channel dimension
            images = images.unsqueeze(1)

            optim.zero_grad()
            logits = model(images)
            loss = discrim(logits, targets)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_dataloader))
        # update tqdm bar value with loss for last epoch
        tqdm.write(f"Epoch {epoch} done, loss: {epoch_loss/len(train_dataloader)}")

    torch.save(model.state_dict(), f"{checkpoint}")

    # plot loss
    plt.plot(train_losses, label="Training loss")
    plt.savefig("reports/figures/loss.png")


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--e", default=5, help="Number of epochs to train for")
@click.option("--checkpoint", default="checkpoint.pt", help="checkpoint  path")
def main(lr, e, checkpoint):
    train(lr, e, checkpoint)


if __name__ == "__main__":
    main()
