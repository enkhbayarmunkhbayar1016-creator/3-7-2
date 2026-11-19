import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

torch.manual_seed(0)

train_set = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor(),
)
test_set = datasets.MNIST('./data', train=False, download=True,
                          transform=transforms.ToTensor(),
)
batch_size = 64
lr = 0.1
epochs = 10
loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

hidden = 128
W1 = torch.randn(hidden, 784)*(2 / 784) ** 0.5
b1 = torch.zeros(hidden)
W2 = torch.randn(10, hidden)*(2 / hidden) ** 0.5
b2 = torch.zeros(10)

def forward(x):
    z = x @ W1.T 
    h = torch.relu(z + b1)
    out = h @ W2.T + b2
    return z, h, out

def softmax(a):
    a = a - a.max(1, keepdim=True).values
    e = torch.exp(a)
    return e / e.sum(1, keepdim=True)

def ce_loss(p, y):
    return -torch.log(p[range(len(y)), y] + 1e-12).mean()

def backward(x, y, z, h, p):
    B = len(y)
    d_out = p.clone()
    d_out[torch.arange(B), y] -= 1
    d_out /= B
    dW2 = d_out.T @ h
    db2 = d_out.sum(0)
    d_h = d_out @ W2
    # relu (х.130): dL/dx_in = dL/dx_out * h'(x_in + b), h'(x) = 1(x >= 0)
    d_z = d_h * ((z + b1) >= 0).float()
    db1 = d_z.sum(0)
    dW1 = d_z.T @ x
    return dW1, db1, dW2, db2

x, y = next(iter(loader))
x = x.view(len(x), -1)
ps = [W1.clone().requires_grad_(), b1.clone().requires_grad_(),
      W2.clone().requires_grad_(), b2.clone().requires_grad_()]
o = torch.relu(x @ ps[0].T + ps[1]) @ ps[2].T + ps[3]
torch.nn.functional.cross_entropy(o, y).backward()
z, h, out = forward(x)
mine = backward(x, y, z, h, softmax(out))
for a, b in zip(mine, ps):
    print('grad diff', (a - b.grad).abs().max().item())
print('initial loss', ce_loss(softmax(out), y).item(), '(ln 10 = 2.30)')

def accuracy(ds):
    ok = 0
    for x, y in DataLoader(ds, batch_size=500):
        out = forward(x.view(len(x), -1))[2]
        ok += (out.argmax(1) == y).sum().item()
    return ok / len(ds)

losses = []
for ep in range(epochs):
    total = 0
    for x, y in loader:
        x = x.view(len(x), -1)
        z, h, out = forward(x)
        p = softmax(out)
        total += ce_loss(p, y).item()
        g = backward(x, y, z, h, p)
        # SGD (х.105-106): theta = theta - lr * dJ/dtheta
        W1 -= lr * g[0]
        b1 -= lr * g[1]
        W2 -= lr * g[2]
        b2 -= lr * g[3]
    losses.append(total / len(loader))
    print('epoch', ep + 1, 'loss', losses[-1])

print('test acc', accuracy(test_set))

torch.save({
    'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
    'epoch': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'hidden': hidden,
    'loss_history': losses,
}, 'checkpoint.pth')
print('saved checkpoint.pth')

ckpt = torch.load('checkpoint.pth')
W1, b1, W2, b2 = ckpt['W1'], ckpt['b1'], ckpt['W2'], ckpt['b2']
print('loaded, epoch', ckpt['epoch'], 'test acc', accuracy(test_set))