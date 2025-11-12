import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def generate_weak_clicks(mask_batch: torch.Tensor,
                         num_pos_clicks: int = 3,
                         num_neg_clicks: int = 3,
                         min_distance: float = 15.0,
                         random_seed: int = None):
    """
    Generate weak labels (positive & negative clicks) from a batch of binary segmentation masks.
    Uses distance transforms and weighted random sampling for better spread.

    Args:
        mask_batch (torch.Tensor): Binary mask tensor [B, 1, H, W], 1 = foreground, 0 = background.
        num_pos_clicks (int): Number of positive clicks per mask.
        num_neg_clicks (int): Number of negative clicks per mask.
        min_distance (float): Minimum pixel distance between clicks.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        batch_positive_clicks: list[list[(x, y)]]
        batch_negative_clicks: list[list[(x, y)]]
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if mask_batch.dim() != 4 or mask_batch.size(1) != 1:
        raise ValueError("Mask must have shape [B, 1, H, W].")

    B, _, H, W = mask_batch.shape
    batch_positive_clicks = []
    batch_negative_clicks = []

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)  # [H*W, 2]

    for b in range(B):
        mask = mask_batch[b, 0].cpu().numpy()

        # Distance transforms. Get distances from each pixel to nearest foreground/background pixel
        # For instance, a pixel centered in the foreground will have a high distance to the background, and vice versa
        # distance_transform_edt will compute the distance between all non-zero pixels to the nearest zero pixel. 
        # It returns the mask with the distances for each pixel.
        #  [[0, 0, 0, 0, 0, 0, 0, 0, 0],        [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #  [0, 1, 1, 1, 1, 1, 1, 0, 0],          [0., 1., 1., 1., 1., 1., 1., 0., 0.],
        #  [0, 1, 1, 1, 1, 1, 1, 0, 0],          [0., 1., 2., 2., 2., 2., 1., 0., 0.],
        #  [0, 1, 1, 1, 1, 1, 1, 0, 0],          [0., 1., 2., 3., 3., 2., 1., 0., 0.],
        #  [0, 1, 1, 1, 1, 1, 1, 0, 0],    //    [0., 1., 2., 3., 3., 2., 1., 0., 0.],
        #  [0, 1, 1, 1, 1, 1, 1, 0, 0],          [0., 1., 2., 2., 2., 2., 1., 0., 0.],
        #  [0, 1, 1, 1, 1, 1, 1, 0, 0],          [0., 1., 1., 1., 1., 1., 1., 0., 0.],
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0],          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0]]          [0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        # As you can see above, this results in higher values (We will turn them into probabilities) to place clicks
        # The same principle applies for the background distance transform, just invert the mask and compute again
        dist_to_bg = distance_transform_edt(mask)         # foreground
        dist_to_fg = distance_transform_edt(1.0 - mask)   # background

        # Weighted probability maps
        # Following the idea of further pixels being higher value, we can compute probabilities
        # for sampling clicks, meaning that clicks are more likely to be far from the boundary
        pos_probs = dist_to_bg.flatten() + 1e-6  # avoid zeros
        pos_probs /= pos_probs.sum()

        neg_probs = dist_to_fg.flatten() + 1e-6
        # Compute max_dist and modify neg_probs with that formula to achieve higher probabilities between the edge of the mask and the edge of the foreground
        # With this, we avoid getting at the edges of the background which would have higher probabilities since they are further away.
        max_dist = np.max(neg_probs)
        neg_probs = dist_to_fg.flatten() * (max_dist - dist_to_fg.flatten())
        neg_probs /= neg_probs.sum()


        # Visualize positive click probabilities
        # #print(f"Grid representation of probabilities for positive clicks (sample {b}):")
        # plt.imshow(pos_probs.reshape(H, W), cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Positive Click Probabilities")
        # plt.savefig(f"img/clicks/pos/pos_click_probs_{b}_testing.png")
        # plt.close()

        # #print(f"Grid representation of probabilities for negative clicks (sample {b}):")
        # plt.imshow(neg_probs.reshape(H, W), cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Negative Click Probabilities")
        # plt.savefig(f"img/clicks/neg/neg_click_probs_{b}_testing.png")
        # plt.close()

        # Helper function: pick clicks with spacing constraint
        def pick_weighted(probs, n_clicks):
            selected = []
            attempts = 0
            max_attempts = n_clicks * 50
            while len(selected) < n_clicks and attempts < max_attempts:
                idx = np.random.choice(len(probs), p=probs)
                x, y = coords[idx]
                if all((x - sx) ** 2 + (y - sy) ** 2 >= min_distance ** 2 for sx, sy in selected):
                    selected.append((int(x), int(y)))
                attempts += 1
            return selected

        pos_clicks = pick_weighted(pos_probs, num_pos_clicks)
        neg_clicks = pick_weighted(neg_probs, num_neg_clicks)

        batch_positive_clicks.append(pos_clicks)
        batch_negative_clicks.append(neg_clicks)

    return batch_positive_clicks, batch_negative_clicks