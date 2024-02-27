import torch
import math
import cv2
import numpy as np
def visualize_and_save_landmarks(image_path, preds, save_path,text=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ensure preds is a NumPy array with shape (98, 2)
    if isinstance(preds, torch.Tensor):
        preds = preds.squeeze(0).numpy()

    # Draw landmarks on the image
    x, y = preds
    cv2.circle(img, (int(x), int(y)), 8, (255, 0, 0), -1)
    if text:
        cv2.putText(img,str(text),(40, 40),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255, 0, 0),thickness=4)
    # Save the image with landmarks
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 3, 'Score maps should be 3-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), -1), 1)

    maxval = maxval.view(scores.size(0), -1)
    idx = idx.view(scores.size(0), -1) + 1

    preds = idx.repeat(1, 2).float()

    preds[:, 0] = (preds[:,  0] - 1) % scores.size(2) + 1
    preds[:, 1] = torch.floor((preds[:,  1] - 1) / scores.size(2)) + 1

    pred_mask = maxval.gt(0).repeat(1, 2).float()
    preds *= pred_mask
    return preds

def decode_preds(output):
    map_width,map_height=output.shape[-2],output.shape[-1]
    coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        hm = output[n]
        px = int(math.floor(coords[n][0]))
        py = int(math.floor(coords[n][1]))
        if (px > 1) and (px < map_width) and (py > 1) and (py < map_height):
            diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
            coords[n] += diff.sign() *.25
    preds = coords.clone()

    # Transform back
    return preds*4 # heatmap is 1/4 to original image


def find_nearest_zero(mask, point):
    h, w = mask.shape
    cx, cy = w // 2, h // 2  # center of the image
    x, y = point
    y=max(0,min(1199,y))
    x=max(0,min(1599,x))
    if mask[y, x] == 0:
        return point

    # Get the direction vector for stepping along the line
    direction = np.array([x - cx, y - cy])
    direction = direction / np.linalg.norm(direction)

    # Step along the line until we hit a zero in the mask
    while 0 <= x < w and 0 <= y < h:
        if mask[int(y), int(x)] == 0:
            return (int(x), int(y))
        
        x += direction[0]
        y += direction[1]

    return (x,y)