import random
import torch
import numpy as np
from t1_dataset import ds
from t2_model import T5

is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"

random.seed(42)
torch.manual_seed(42)


def train():
    trn_dl = torch.utils.data.DataLoader(
        ds, batch_size=16, shuffle=True, collate_fn=ds.collate_fn
    )

    myT5 = T5().to(device)
    myT5.num_params()

    num_epochs = 10
    lr = 0.0001

    opt = torch.optim.Adam(myT5.parameters(), lr=lr)

    for epoch in range(num_epochs):
        myT5.train()
        total_loss = 0.0
        for idx, batch in enumerate(trn_dl):
            c = batch["contx"].to(device)
            x = batch["input"].to(device)
            y = batch["label"].to(device)
            p = myT5(c, x)

            p = p.view(-1, p.size(-1))
            y = y.view(-1)
            l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)
            if idx % 1000 == 0:
                print(f"Loss({epoch}_{idx}): {l.item():.4f}")

            l.backward()
            opt.step()
            opt.zero_grad()
            total_loss += l.item()

        # Print average loss for the epoch
        torch.save(myT5.state_dict(), f"weights_{epoch}_{idx}.pt")
        avg_train_loss = total_loss / len(trn_dl)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    # Save your model's state_dict locally
    torch.save(myT5.state_dict(), "final_weights.pth")


if __name__ == "__main__":
    train()
