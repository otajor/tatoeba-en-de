# %%
import torch
import numpy as np
from t1_dataset import trn_dl, tst_dl, ds
import torch
import random
import wandb
import time

is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"


EMBD = 128
HEAD = 4
BLKS = 8
DROP = 0.1
SQNZ = 512
VOCB = 10000


# %%
class Attention(torch.nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.out_proj = torch.nn.Linear(EMBD, EMBD)
        self.register_buffer(
            "mask", torch.tril(torch.ones(SQNZ, SQNZ).view(1, 1, SQNZ, SQNZ))
        )

    def forward(self, qry, key, val):
        Q_B, Q_S, _ = qry.shape
        K_B, K_S, _ = key.shape
        V_B, V_S, _ = val.shape
        EMBD_HEAD = int(EMBD / HEAD)

        qry = qry.reshape(Q_B, Q_S, HEAD, EMBD_HEAD).transpose(1, 2)
        key = key.reshape(K_B, K_S, HEAD, EMBD_HEAD).transpose(1, 2)
        val = val.reshape(V_B, V_S, HEAD, EMBD_HEAD).transpose(1, 2)

        msk = self.mask[:, :, :Q_S, :Q_S] == 0
        att = qry @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(EMBD_HEAD))
        att = att if self.is_causal == False else att.masked_fill(msk, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        out = (att @ val).transpose(1, 2).reshape(Q_B, Q_S, EMBD)
        return self.out_proj(out)


class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = torch.nn.Linear(EMBD, EMBD * 4)
        self.relu = torch.nn.ReLU()
        self.c_proj = torch.nn.Linear(EMBD * 4, EMBD)
        self.drop = torch.nn.Dropout(DROP)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(EMBD)
        self.qkv = torch.nn.Linear(EMBD, EMBD * 3)
        self.attn = Attention()
        self.ln_2 = torch.nn.LayerNorm(EMBD)
        self.ffww = FeedForward()

    def forward(self, x):
        q, k, v = self.qkv(self.ln_1(x)).split(EMBD, dim=-1)
        x = x + self.attn(q, k, v)
        x = x + self.ffww(self.ln_2(x))
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(EMBD, EMBD * 3)
        self.qry = torch.nn.Linear(EMBD, EMBD)
        self.key = torch.nn.Linear(EMBD, EMBD)
        self.val = torch.nn.Linear(EMBD, EMBD)
        self.c_att = Attention(is_causal=True)
        self.x_attn = Attention()
        self.ffww = FeedForward()

    def forward(self, src, tgt):
        q, k, v = self.qkv(tgt).split(EMBD, dim=-1)
        tgt = tgt + self.c_att(q, k, v)

        qry = self.qry(tgt)
        key = self.key(src)
        val = self.val(src)

        tgt = tgt + self.x_attn(qry, key, val)
        tgt = tgt + self.ffww(tgt)
        return tgt


class T5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embd = torch.nn.Embedding(VOCB, EMBD)
        self.pos_embd = torch.nn.Embedding(SQNZ, EMBD)
        self.enc_blks = torch.nn.ModuleList([EncoderBlock() for _ in range(BLKS)])
        self.dec_blks = torch.nn.ModuleList([DecoderBlock() for _ in range(BLKS)])
        self.vocab = torch.nn.Linear(EMBD, VOCB)

    def forward(self, src, tgt):
        src = self.tok_embd(src)
        src = src + self.pos_embd(torch.arange(src.size(1), device=device))
        for blk in self.enc_blks:
            src = blk(src)

        tgt = self.tok_embd(tgt)
        tgt = tgt + self.pos_embd(torch.arange(tgt.size(1), device=device))
        for blk in self.dec_blks:
            tgt = blk(src, tgt)
        tgt = self.vocab(tgt)
        return tgt

    def num_params(self):
        gpt_params = sum(p.numel() for p in self.parameters())
        emb_params = self.tok_embd.weight.numel()
        print(f"Total Parameters: {gpt_params} | Embedding: {emb_params}")
        return {"gpt_params": gpt_params, "emb_params": emb_params}

    def translate(self, src, num=20):
        self.eval()
        tgt = torch.tensor([[2]], device=device)
        for _ in range(num):
            with torch.no_grad():
                out = self(src, tgt)
                out = out[:, -1, :]
                nxt = torch.argmax(out, dim=-1, keepdim=True)
                if nxt.item() == 3:
                    break
                tgt = torch.cat((tgt, nxt), dim=1)
        self.train()
        return tgt
#%%

is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"


random.seed(42)
torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(0.5, device=None)
myT5 = T5().to(device)
myT5.num_params()

num_epochs = 10
lr = 0.0001

wandb.init(project="en_de_tatoeba_otm")
config = wandb.config
config.emb_size = EMBD
config.max_seq_len = SQNZ
config.vocab_size = VOCB
config.num_epochs = num_epochs
config.lr = lr


opt = torch.optim.Adam(myT5.parameters(), lr=lr)

for epoch in range(num_epochs):
    org = "Hello my name is Bes and I work in the field of AI."
    src = torch.tensor([ds.english_sp.encode(org)]).to(device)
    trs = myT5.translate(src)
    print(f"{org} - {ds.german_sp.decode(trs.tolist()[0])}")
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


# %%
