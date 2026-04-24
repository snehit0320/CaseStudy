import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Learnable scores that become gates after sigmoid.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_features
        bound = 1 / fan_in**0.5
        nn.init.uniform_(self.bias, -bound, bound)
        # Start partially open so L1 can push many gates to near-zero faster.
        nn.init.normal_(self.gate_scores, mean=-1.5, std=0.1)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def l1_gate_penalty(self) -> torch.Tensor:
        penalties = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                penalties.append(module.gates().abs().sum())
        return torch.stack(penalties).sum()

    def sparsity(self, threshold: float = 1e-2) -> float:
        total = 0
        pruned = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    g = module.gates()
                    total += g.numel()
                    pruned += (g < threshold).sum().item()
        return 100.0 * pruned / max(1, total)


def get_dataloaders(batch_size: int = 128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lambda_l1: float):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    total = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        ce_loss = F.cross_entropy(logits, labels)
        reg_loss = model.l1_gate_penalty()
        loss = ce_loss + lambda_l1 * reg_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_ce_loss += ce_loss.item() * labels.size(0)
        total_sparse_loss += reg_loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return (
        total_loss / total,
        total_ce_loss / total,
        total_sparse_loss / total,
        100.0 * correct / total,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return 100.0 * correct / total


def run_experiment(lambda_l1: float, epochs: int = 12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=128)
    model = PrunableMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n=== Lambda: {lambda_l1} ===", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, ce_loss, sparse_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, lambda_l1
        )
        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d} | total_loss={train_loss:.4f} "
            f"| ce_loss={ce_loss:.4f} | sparsity_loss={sparse_loss:.4f} "
            f"| train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%"
            ,
            flush=True,
        )

    final_sparsity_1e2 = model.sparsity(threshold=1e-2)
    final_sparsity_5e2 = model.sparsity(threshold=5e-2)
    final_sparsity_1e1 = model.sparsity(threshold=1e-1)
    final_test_acc = evaluate(model, test_loader, device)
    print(f"Final sparsity (<1e-2 gates): {final_sparsity_1e2:.2f}%", flush=True)
    print(f"Final sparsity (<5e-2 gates): {final_sparsity_5e2:.2f}%", flush=True)
    print(f"Final sparsity (<1e-1 gates): {final_sparsity_1e1:.2f}%", flush=True)
    print(f"Final test accuracy: {final_test_acc:.2f}%", flush=True)
    return final_sparsity_1e2, final_sparsity_5e2, final_sparsity_1e1, final_test_acc


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting self-pruning training on {device_name}", flush=True)
    print("This run compares 3 lambda values: low, medium, high.", flush=True)

    # Compare low / medium / high lambda to show trade-off.
    lambdas = [1e-5, 5e-5, 1e-4]
    results = {}
    for lam in lambdas:
        s1, s5, s10, acc = run_experiment(lambda_l1=lam, epochs=12)
        results[lam] = (s1, s5, s10, acc)

    print("\n=== Summary ===", flush=True)
    print("lambda      s@1e-2(%)  s@5e-2(%)  s@1e-1(%)  test_acc(%)", flush=True)
    print("----------------------------------------------------------", flush=True)
    for lam, (s1, s5, s10, a) in results.items():
        print(f"{lam:.0e}       {s1:8.2f}   {s5:8.2f}   {s10:8.2f}   {a:10.2f}", flush=True)
