import torch

mean = torch.tensor([0.1, 0.2], requires_grad=True)
query = torch.tensor([0.3, 0.4])

cov = torch.tensor([[0.5, 0.2], [0.2, 0.8]], requires_grad=True)

G = torch.exp(
    -0.5
    * torch.matmul(torch.matmul((query - mean), torch.inverse(cov)), (query - mean).t())
)

G.backward()
print(G.item())
print(mean.grad)
print(cov.grad)

mean = torch.tensor([0.1, 0.2])
query = torch.tensor([0.3, 0.4])

cov = torch.tensor([0.5, 0.2, 0.2, 0.8])

x = query - mean

det = cov[0] * cov[3] - cov[1] * cov[2]
cov = torch.tensor([[cov[3], -cov[1]], [-cov[2], cov[0]]]) / det
tmp = torch.matmul(x, cov)
val = torch.exp(-0.5 * torch.matmul(tmp, x.t())).item()
print(val)

grad_mean = val * tmp
print(grad_mean)

grad_cov = val * torch.matmul(tmp.unsqueeze(1), tmp.unsqueeze(0)) * 0.5
print(grad_cov)
