import torch
import torch.nn as nn
import numpy as np
import json
import os

import matplotlib.pyplot as plt

from clicks_generator import *

def plot_segmentation_metrics(log, save_path='segmentation_metrics.png'):
    
    metrics = list(log.keys())
    num_metrics = len(metrics)
    num_cols = 4
    num_rows = (num_metrics + num_cols - 1) // num_cols

    plt.figure(figsize=(16, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(log[metric], marker='o', label=metric)
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics plot saved to: {save_path}")


import matplotlib.pyplot as plt

def visualize_segmentation(image, gt_mask, pred_mask, pos_clicks=None, neg_clicks=None, save_path="segmentation_example.png"):
    """
    image: [C, H, W] tensor (e.g., RGB)
    gt_mask: [H, W] tensor (binary or multi-class)
    pred_mask: [H, W] tensor (binary or multi-class)
    pos_clicks, neg_clicks: list of (x, y) tuples
    save_path: path to save the figure
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    gt_np = gt_mask.cpu().numpy()
    pred_np = pred_mask.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    if pos_clicks:
        for x, y in pos_clicks:
            axs[0].plot(x, y, 'g+', markersize=10)
    if neg_clicks:
        for x, y in neg_clicks:
            axs[0].plot(x, y, 'r_', markersize=10)

    axs[1].imshow(gt_np, cmap='gray')
    axs[1].set_title("Ground Truth")

    axs[2].imshow(pred_np, cmap='gray')
    axs[2].set_title("Predicted Mask")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def dice_score(y_pred, y_true, smooth=1e-6):
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    intersection = (y_pred_flat * y_true_flat).sum()
    return (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)

def iou_score(y_pred, y_true, smooth=1e-6):
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    intersection = (y_pred_flat * y_true_flat).sum()
    union = y_pred_flat.sum() + y_true_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def sensitivity(y_pred, y_true):
    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()
    return TP / (TP + FN + 1e-6)

def specificity(y_pred, y_true):
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    return TN / (TN + FP + 1e-6)

def train_seg(model, optimizer, loss_fn, train_loader, test_loader, val_loader, num_epochs=10, device=None, log_path='seg_log.json'):
    
    device = next(model.parameters()).device if device is None else device
    print("Training will be carried out on:", device)

    log = {
        'train_loss': [], 'train_accuracy': [], 'train_dice': [], 'train_iou': [], 'train_sensitivity': [], 'train_specificity': [],
        'val_loss': [], 'val_accuracy': [], 'val_dice': [], 'val_iou': [], 'val_sensitivity': [], 'val_specificity': [],
        'test_loss': [], 'test_accuracy': [], 'test_dice': [], 'test_iou': [], 'test_sensitivity': [], 'test_specificity': []
    }

    threshold = 0.5

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = []
        correct_pixels = 0
        total_pixels = 0
        dice_total, iou_total, sens_total, spec_total = 0, 0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            pred_bin = (torch.sigmoid(output) > threshold).float()
            correct_pixels += (pred_bin == target).sum().item()
            total_pixels += torch.numel(target)

            dice_total += dice_score(pred_bin, target)
            iou_total += iou_score(pred_bin, target)
            sens_total += sensitivity(pred_bin, target)
            spec_total += specificity(pred_bin, target)

        log['train_loss'].append(np.mean(train_loss))
        log['train_accuracy'].append(correct_pixels / total_pixels)
        log['train_dice'].append(dice_total / len(train_loader))
        log['train_iou'].append(iou_total / len(train_loader))
        log['train_sensitivity'].append(sens_total / len(train_loader))
        log['train_specificity'].append(spec_total / len(train_loader))

        # Validation
        model.eval()
        val_loss = []
        correct_pixels = 0
        total_pixels = 0
        dice_total, iou_total, sens_total, spec_total = 0, 0, 0, 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                val_loss.append(loss.item())

                pred_bin = (torch.sigmoid(output) > threshold).float()
                correct_pixels += (pred_bin == target).sum().item()
                total_pixels += torch.numel(target)

                dice_total += dice_score(pred_bin, target)
                iou_total += iou_score(pred_bin, target)
                sens_total += sensitivity(pred_bin, target)
                spec_total += specificity(pred_bin, target)

        log['val_loss'].append(np.mean(val_loss))
        log['val_accuracy'].append(correct_pixels / total_pixels)
        log['val_dice'].append(dice_total / len(val_loader))
        log['val_iou'].append(iou_total / len(val_loader))
        log['val_sensitivity'].append(sens_total / len(val_loader))
        log['val_specificity'].append(spec_total / len(val_loader))

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {log['train_loss'][-1]:.4f} | Accuracy: {log['train_accuracy'][-1]*100:.2f}% | Dice: {log['train_dice'][-1]:.4f} | IoU: {log['train_iou'][-1]:.4f} | Sensitivity: {log['train_sensitivity'][-1]:.4f} | Specificity: {log['train_specificity'][-1]:.4f}")
        print(f"Val   Loss: {log['val_loss'][-1]:.4f} | Accuracy: {log['val_accuracy'][-1]*100:.2f}% | Dice: {log['val_dice'][-1]:.4f} | IoU: {log['val_iou'][-1]:.4f} | Sensitivity: {log['val_sensitivity'][-1]:.4f} | Specificity: {log['val_specificity'][-1]:.4f}")

    # Final test evaluation
    model.eval()
    test_loss = []
    correct_pixels = 0
    total_pixels = 0
    dice_total, iou_total, sens_total, spec_total = 0, 0, 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss.append(loss.item())

            pred_bin = (torch.sigmoid(output) > threshold).float()
            correct_pixels += (pred_bin == target).sum().item()
            total_pixels += torch.numel(target)

            dice_total += dice_score(pred_bin, target)
            iou_total += iou_score(pred_bin, target)
            sens_total += sensitivity(pred_bin, target)
            spec_total += specificity(pred_bin, target)

    log['test_loss'].append(np.mean(test_loss))
    log['test_accuracy'].append(correct_pixels / total_pixels)
    log['test_dice'].append(dice_total / len(test_loader))
    log['test_iou'].append(iou_total / len(test_loader))
    log['test_sensitivity'].append(sens_total / len(test_loader))
    log['test_specificity'].append(spec_total / len(test_loader))

    print(f"\nFinal Test Evaluation")
    print(f"Test  Loss: {log['test_loss'][-1]:.4f} | Accuracy: {log['test_accuracy'][-1]*100:.2f}% | Dice: {log['test_dice'][-1]:.4f} | IoU: {log['test_iou'][-1]:.4f} | Sensitivity: {log['test_sensitivity'][-1]:.4f} | Specificity: {log['test_specificity'][-1]:.4f}")

    # Save log
    log_clean = {
        k: [float(v) if torch.is_tensor(v) else float(v) if isinstance(v, np.ndarray) else v
            for v in vals]
        for k, vals in log.items()
    }

    with open(log_path, 'w') as f:
        json.dump(log_clean, f, indent=4)

    print(f"Training log saved to {os.path.abspath(log_path)}")

    # Save visualization
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_bin = (torch.sigmoid(output) > threshold).float()

            visualize_segmentation(
                image=data[0],
                gt_mask=target[0, 0],
                pred_mask=pred_bin[0, 0],
                save_path="final_validation_result.png"
            )
            break

    return log



# Calc BCELoss for the given points
def weak_loss(pred, pos_clicks, neg_clicks):
    loss = 0
    for b in range(pred.size(0)):
        for x, y in pos_clicks[b]:
            loss += F.binary_cross_entropy_with_logits(pred[b, 0, y, x], torch.tensor(1.0).to(pred.device))
        for x, y in neg_clicks[b]:
            loss += F.binary_cross_entropy_with_logits(pred[b, 0, y, x], torch.tensor(0.0).to(pred.device))
    return loss / (len(pos_clicks) + len(neg_clicks))


def train_seg_weak_supervision(model, optimizer, train_loader, test_loader, num_epochs=10, 
                               num_pos_clicks=3,
                               num_neg_clicks=3,
                               device=None, log_path='seg_log.json'):
    
    device = next(model.parameters()).device if device is None else device
    print("Training will be carried out on:", device,f"using weak supervision with {num_pos_clicks} positive clicks and {num_neg_clicks} negative clicks.")

    log = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_dice': [],
        'val_iou': [],
        'val_sensitivity': [],
        'val_specificity': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            pos_clicks, neg_clicks = generate_weak_clicks(target,num_pos_clicks=num_pos_clicks,
                                                                num_neg_clicks=num_neg_clicks)
            loss = weak_loss(output, pos_clicks, neg_clicks)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        correct_pixels = 0
        total_pixels = 0
        dice_total, iou_total, sens_total, spec_total = 0, 0, 0, 0
        threshold = 0.5

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                pos_clicks, neg_clicks = generate_weak_clicks(target,num_pos_clicks=num_pos_clicks,
                                                                num_neg_clicks=num_neg_clicks)
                loss = weak_loss(output, pos_clicks, neg_clicks)

                val_loss.append(loss.item())

                pred_bin = (torch.sigmoid(output) > threshold).float()
                correct_pixels += (pred_bin == target).sum().item()
                total_pixels += torch.numel(target)

                dice_total += dice_score(pred_bin, target)
                iou_total += iou_score(pred_bin, target)
                sens_total += sensitivity(pred_bin, target)
                spec_total += specificity(pred_bin, target)
        model.eval()
        # Logging
        log['train_loss'].append(np.mean(train_loss))
        log['val_loss'].append(np.mean(val_loss))
        log['val_accuracy'].append(correct_pixels / total_pixels)
        log['val_dice'].append(dice_total / len(test_loader))
        log['val_iou'].append(iou_total / len(test_loader))
        log['val_sensitivity'].append(sens_total / len(test_loader))
        log['val_specificity'].append(spec_total / len(test_loader))

        print(f"Epoch {epoch+1}/{num_epochs}; Train Loss: {log['train_loss'][-1]:.4f}; Val Loss: {log['val_loss'][-1]:.4f}")
        print(f"Val Accuracy: {log['val_accuracy'][-1]*100:.2f}% | Dice: {log['val_dice'][-1]:.4f} | IoU: {log['val_iou'][-1]:.4f} | Sensitivity: {log['val_sensitivity'][-1]:.4f} | Specificity: {log['val_specificity'][-1]:.4f}")

    # Convert all tensor values to floats
    log_clean = {
        k: [float(v) if torch.is_tensor(v) else float(v) if isinstance(v, np.ndarray) else v
            for v in vals]
        for k, vals in log.items()
    }

    # Save to disk
    with open(log_path, 'w') as f:
        json.dump(log_clean, f, indent=4)

    print(f"Training log saved to {os.path.abspath(log_path)}")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_bin = (torch.sigmoid(output) > threshold).float()

            # Generate simulated clicks for visualization
            pos_clicks, neg_clicks = generate_weak_clicks(target,num_pos_clicks=num_pos_clicks,
                                                                num_neg_clicks=num_neg_clicks)

            # Save visualization for first sample in batch
            visualize_segmentation(
                image=data[0],
                gt_mask=target[0, 0],
                pred_mask=pred_bin[0, 0],
                pos_clicks=pos_clicks[0],
                neg_clicks=neg_clicks[0],
                save_path="final_validation_result.png"
            )
            break  # Only visualize one batch
    return log