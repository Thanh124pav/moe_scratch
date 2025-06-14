import torch
import torch.nn as nn


# Giả lập mô hình và dataloader
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

def get_dataloader(num_batches, batch_size):
    for _ in range(num_batches):
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 1)
        yield {'input': x, 'target': y}

# Khởi tạo wandb


model = DummyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

num_epochs = 3
train_batches = 20
eval_batches = 5
batch_size = 8

for epoch in range(num_epochs):
    # ----- TRAINING -----
    model.train()
    total_train_loss = 0
    num_train_steps = 0
    for i, batch in enumerate(get_dataloader(train_batches, batch_size)):
        optimizer.zero_grad()
        output = model(batch['input'])
        loss = loss_fn(output, batch['target'])
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        num_train_steps += 1
        # Log loss từng step (nếu muốn)
        wandb.log({"train/loss": loss.item(), "train/step": i + epoch * train_batches})
    avg_train_loss = total_train_loss / num_train_steps
    wandb.log({"train/avg_loss": avg_train_loss, "epoch": epoch})
    print(f"Epoch {epoch}, Train avg loss: {avg_train_loss:.4f}")

    # ----- EVALUATION -----
    model.eval()
    total_eval_loss = 0
    num_eval_steps = 0
    with torch.no_grad():
        for j, batch in enumerate(get_dataloader(eval_batches, batch_size)):
            output = model(batch['input'])
            eval_loss = loss_fn(output, batch['target'])
            total_eval_loss += eval_loss.item()
            num_eval_steps += 1
            # Log loss từng eval step (nếu muốn)
            wandb.log({"eval/loss": eval_loss.item(), "eval/step": j + epoch * eval_batches})
    avg_eval_loss = total_eval_loss / num_eval_steps
    wandb.log({"eval/avg_loss": avg_eval_loss, "epoch": epoch})
    print(f"Epoch {epoch}, Eval avg loss: {avg_eval_loss:.4f}")

wandb.finish() 