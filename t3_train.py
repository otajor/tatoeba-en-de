import random
import wandb
import time
import torch
import numpy as np
from t1_dataset import trn_dl, tst_dl, ds
from t2_model import T5

is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"


random.seed(42)
torch.manual_seed(42)
myT5 = T5().to(device)
myT5.num_params()

num_epochs = 10
lr = 0.0001

EMBD = 128
HEAD = 4
BLKS = 8
DROP = 0.1
SQNZ = 512
VOCB = 10000
wandb.init(project="en_de_tatoeba_otm")
config = wandb.config
config.emb_size = EMBD
config.max_seq_len = SQNZ
config.vocab_size = VOCB
config.num_epochs = num_epochs
config.lr = lr


opt = torch.optim.Adam(myT5.parameters(), lr=lr)

for epoch in range(num_epochs):
    myT5.train()
    total_loss = 0.0
    for idx, batch in enumerate(trn_dl):
        start_time = time.time()

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

        # Calculate the time taken for the batch
        batch_time = time.time() - start_time
        # Calculate the max sequence length from the batch
        max_seq_length = max(c.size(1), x.size(1))
        wandb.log({
            "Loss": l.item(),
            "Batch Time": batch_time,
            "Max Sequence Length": max_seq_length
        })



    # Print average loss for the epoch
    torch.save(myT5.state_dict(), f"weights_{epoch}_{idx}.pt")
    avg_train_loss = total_loss / len(trn_dl)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    # Validation loop
    myT5.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(tst_dl):
            c = batch["contx"].to(device)
            x = batch["input"].to(device)
            y = batch["label"].to(device)
            p = myT5(c, x)

            p = p.view(-1, p.size(-1))
            y = y.view(-1)
            l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)

            total_val_loss += l.item()

    avg_val_loss = total_val_loss / len(tst_dl)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    wandb.log(
        {
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
        }
    )

# Save your model's state_dict locally
torch.save(myT5.state_dict(), "final_weights.pth")


# Upload the saved model file to wandb
wandb.save("tiny_stories.pth")
# save to wandb

wandb.finish()