import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layers = 6
dropout = 0.2


torch.manual_seed(1337)

# data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique chars
chars = sorted(list(set(list(text))))
vocab_size = len(chars)

# mappings
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[ix: ix+block_size] for ix in ixs])
    y = torch.stack([data[ix+1: ix+block_size+1] for ix in ixs])
    x, y = x.to(device=device), y.to(device=device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

####################### Model Code ######################
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # C: n_embd

        q = self.query(x) # B, T, C: head_size
        k = self.key(x) # B, T, C: head_size
        v = self.value(x) # B, T, C: head_size
        head_size = q.shape[-1]

        wei = q @ k.transpose(-2, -1)*head_size**-0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B,T,T
        wei = F.softmax(wei, dim=-1) # B,T,T
        wei = self.dropout(wei) # B,T,T

        out = wei @ v # B,T,C: head_size
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape:= B,T,C:n_embd

        # h(x).shape:= B,T,C: head_size
        out = torch.cat([h(x) for h in self.heads], dim=-1) # h(x).shape:= B,T,C: head_size*num_heads
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        ])

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positiona_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_head) for _ in range(n_layers)
        ])
        self.lm_head = nn.Sequential(*[
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, vocab_size)
        ])

    def forward(self, x, targets=None):
        B, T = x.shape

        # 
        tok_emb = self.token_embedding_table(x) # B, T, C:n_embd
        pos_emb = self.positiona_embedding_table(torch.arange(T, device=device)) # T, C:n_embd
        x = tok_emb + pos_emb
        x = self.blocks(x)

        logits = self.lm_head(x) # B, T, C: vocab_size
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens):
        # x: B, T
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits, loss = self(x_cond) # B, T, C: vocab_size
            # focus only on the last timestep
            logits = logits[:, -1, :] # B, C: vocab_size
            probs = F.softmax(logits, dim=1) # B, C: vocab_size
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat((x, x_next), dim=1) # B, T+1
        return x

model = LLM()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

total = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable_params:,}")

exit()
# Training

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
else:
    losses = estimate_loss(model)
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


model.eval()
with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))