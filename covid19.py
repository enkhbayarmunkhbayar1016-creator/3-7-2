import os
import urllib.request
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/new_cases.csv'
if not os.path.exists('covid.csv'):
    urllib.request.urlretrieve(url, 'covid.csv')

df = pd.read_csv('covid.csv')
series = df['World'].fillna(0)
series = series.rolling(7, min_periods=1).mean().values.astype('float32')

n_train = int(len(series) * 0.8)
mn, mx = series[:n_train].min(), series[:n_train].max()
data = ((series - mn) / (mx - mn)).astype('float32')

win = 14


def make_xy(d):
    x, y = [], []
    for i in range(len(d) - win):
        x.append(d[i:i + win])
        y.append(d[i + win])
    return (torch.tensor(np.array(x)).unsqueeze(-1),
            torch.tensor(np.array(y)).unsqueeze(-1))


trainx, trainy = make_xy(data[:n_train])
testx, testy = make_xy(data[n_train - win:])

D = 1
H = 16


def init(*shape):
    return torch.randn(*shape) * 0.3


P = {
    'Wu': init(H, H), 'Uu': init(H, D), 'bu': torch.zeros(H),
    'Wr': init(H, H), 'Ur': init(H, D), 'br': torch.zeros(H),
    'Wh': init(H, H), 'Uh': init(H, D), 'bh': torch.zeros(H),
    'Wy': init(1, H), 'by': torch.zeros(1),
}


def forward(x):
    B, T, _ = x.shape
    h = torch.zeros(B, H)
    cache = []
    for t in range(T):
        xt = x[:, t]
        u = torch.sigmoid(h @ P['Wu'].T + xt @ P['Uu'].T + P['bu'])
        r = torch.sigmoid(h @ P['Wr'].T + xt @ P['Ur'].T + P['br'])
        hc = torch.tanh((r * h) @ P['Wh'].T + xt @ P['Uh'].T + P['bh'])
        h_new = (1 - u) * h + u * hc
        cache.append((xt, h, u, r, hc))
        h = h_new
    y = h @ P['Wy'].T + P['by']
    return y, h, cache


def backward(y, t, h_last, cache):
    B = len(y)
    g = {k: torch.zeros_like(v) for k, v in P.items()}
    dy = (y - t) / B
    g['Wy'] = dy.T @ h_last
    g['by'] = dy.sum(0)
    dh = dy @ P['Wy']
    
    for xt, h_prev, u, r, hc in reversed(cache):
        du = dh * (hc - h_prev)
        dhc = dh * u
        dh_prev = dh * (1 - u)

        da_c = dhc * (1 - hc ** 2)
        g['Wh'] += da_c.T @ (r * h_prev)
        g['Uh'] += da_c.T @ xt
        g['bh'] += da_c.sum(0)
        d_rh = da_c @ P['Wh']
        dr = d_rh * h_prev
        dh_prev = dh_prev + d_rh * r

        da_r = dr * r * (1 - r)
        g['Wr'] += da_r.T @ h_prev
        g['Ur'] += da_r.T @ xt
        g['br'] += da_r.sum(0)
        dh_prev = dh_prev + da_r @ P['Wr']

        da_u = du * u * (1 - u)
        g['Wu'] += da_u.T @ h_prev
        g['Uu'] += da_u.T @ xt
        g['bu'] += da_u.sum(0)
        dh_prev = dh_prev + da_u @ P['Wu']

        dh = dh_prev
    return g


def loss_fn(y, t):
    return 0.5 * ((y - t) ** 2).mean().item()


xb, yb = trainx[:8], trainy[:8]
y, h, cache = forward(xb)
mine = backward(y, yb, h, cache)
Q = {k: v.clone().requires_grad_() for k, v in P.items()}
P_old = P
P = Q
y2, _, _ = forward(xb)
(0.5 * ((y2 - yb) ** 2).mean()).backward()
P = P_old
for k in P:
    print(k, 'grad diff', (mine[k] - Q[k].grad).abs().max().item())

lr = 0.5
clip = 1.0
batch = 32
train_losses, test_losses = [], []
for ep in range(150):
    perm = torch.randperm(len(trainx))
    total = 0
    for i in range(0, len(perm), batch):
        idx = perm[i:i + batch]
        y, h, cache = forward(trainx[idx])
        total += loss_fn(y, trainy[idx])
        g = backward(y, trainy[idx], h, cache)
        norm = torch.sqrt(sum((v ** 2).sum() for v in g.values()))
        if norm >= clip:
            for k in g:
                g[k] = g[k] * clip / norm
        for k in P:
            P[k] -= lr * g[k]
    train_losses.append(total / (len(perm) / batch))
    test_losses.append(loss_fn(forward(testx)[0], testy))
    if (ep + 1) % 10 == 0:
        print('epoch', ep + 1, 'train', train_losses[-1], 'test', test_losses[-1])

pred = forward(testx)[0].squeeze().numpy() * (mx - mn) + mn
real = testy.squeeze().numpy() * (mx - mn) + mn
print('test mse (scaled):', 2 * test_losses[-1])

torch.save({'params': P, 'hidden': H, 'window': win, 'min': mn, 'max': mx,
            'train_loss': train_losses, 'test_loss': test_losses}, 'gru_covid.pth')

plt.figure()
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('covid_loss.png')

plt.figure(figsize=(10, 5))
plt.plot(real, label='real')
plt.plot(pred, label='GRU prediction')
plt.title('Covid19 new cases (World) - test set')
plt.xlabel('day')
plt.ylabel('cases')
plt.legend()
plt.savefig('covid_test.png')
plt.show()