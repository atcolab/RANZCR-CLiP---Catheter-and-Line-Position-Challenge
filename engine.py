import torch
import numpy as np
from apex import amp
from tqdm import tqdm

def train_loop_fn(model, loader, optimizer, loss_func, device):
    model.train()

    TRAIN_LOSS = []

    bar = tqdm(enumerate(loader), total=len(loader))

    for step, (data, target) in bar:
        data = data.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(data)
        probs = torch.sigmoid(outputs)
        loss = loss_func(outputs, target)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        TRAIN_LOSS.append(loss.item())
        smooth_loss = np.mean(TRAIN_LOSS[-30:])
        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

        optimizer.step()
  
    avg_train_loss = np.mean(TRAIN_LOSS)

    return avg_train_loss

def val_loop_fn(model, loader, optimizer, loss_func, device):

    model.eval()

    VAL_LOSS = []
    PREDS = []
    TARGS = []

    bar = tqdm(enumerate(loader), total=len(loader))

    with torch.no_grad():
        for step, (data, target) in bar:
            data = data.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(data)
            probs = torch.sigmoid(outputs)
            loss = loss_func(outputs, target)

            VAL_LOSS.append(loss.item())

            smooth_loss = np.mean(VAL_LOSS[-30:])
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

            PREDS += [target.detach().cpu()]
            TARGS += [probs.detach()]

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGS = torch.cat(TARGS).cpu().numpy()

    avg_val_loss = np.mean(VAL_LOSS)

    return avg_val_loss, PREDS, TARGS