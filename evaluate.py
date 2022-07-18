import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        low, high_true = batch['low'], batch['high']
        # move lows and labels to correct device and type
        low = low.to(device=device, dtype=torch.float32)
        high_true = high_true.to(device=device, dtype=torch.long)
        high_true = F.one_hot(high_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the high
            high_pred = net(low)

            # convert to one-hot format
            if net.n_classes == 1:
                high_pred = (F.sigmoid(high_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(high_pred, high_true, reduce_batch_first=False)
            else:
                high_pred = F.one_hot(high_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(high_pred[:, 1:, ...], high_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
