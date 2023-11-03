import torch
import torch.nn.functional as functional

# TODO : add dataloader
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = functional.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# TODO : add dataloader
@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs_losses = model(data), []
    keys = ['train', 'val', 'test']
    for key in keys:
        mask = data[f'{key}_mask']
        predictions = logits[mask].max(1)[1]
        acc = predictions.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs_losses.append(acc)
        loss = functional.nll_loss(logits[mask], data.y[mask])
        accs_losses.append(loss)
    return accs_losses