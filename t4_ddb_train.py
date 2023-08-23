import random
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import os
from t1_dataset import ds
from t2_model import T5

random.seed(42)
torch.manual_seed(42)


def train(rank, world_size):
    # Initialize the distributed environment
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=16, sampler=sampler, collate_fn=ds.collate_fn
    )

    myT5 = T5().to(rank)
    myT5 = DistributedDataParallel(myT5, device_ids=[rank])

    num_epochs = 10
    lr = 0.0001

    opt = torch.optim.Adam(myT5.parameters(), lr=lr)

    for epoch in range(num_epochs):
        myT5.train()
        total_loss = 0.0
        for idx, batch in enumerate(dl):
            c = batch["contx"].to(rank)
            x = batch["input"].to(rank)
            y = batch["label"].to(rank)
            p = myT5(c, x)

            p = p.view(-1, p.size(-1))
            y = y.view(-1)
            l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)
            if idx % 200 == 0:
                print(f"Loss({epoch}_{idx}): {l.item():.4f}")

            l.backward()
            opt.step()
            opt.zero_grad()
            total_loss += l.item()

        # Print average loss for the epoch
        torch.save(myT5.state_dict(), f"weights_{epoch}_{idx}.pt")
        avg_train_loss = total_loss / len(dl)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    # Save your model's state_dict locally
    torch.save(myT5.state_dict(), "final_weights.pth")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
