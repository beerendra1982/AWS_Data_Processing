import argparse
import os
import torch
from torch.utils.data import DataLoader
# from your_dataset import SatelliteDataset
# from your_model import MyModel

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_data = SatelliteDataset(os.environ["SM_CHANNEL_TRAIN"])
    val_data   = SatelliteDataset(os.environ["SM_CHANNEL_VAL"])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Save model
    model_dir = os.environ["SM_MODEL_DIR"]
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
