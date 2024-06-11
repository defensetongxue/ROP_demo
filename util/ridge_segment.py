from PIL import Image, ImageFont, ImageDraw
import numpy as np

def k_max_values_and_indices(scores, k,r=100,threshold=0.0):
    # Flatten the array and get the indices of the top-k values

    preds_list = []
    maxvals_list = []

    for _ in range(k):
        idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)

        maxval = scores[idx]
        if maxval<threshold:
            break
        maxvals_list.append(float(maxval))
        preds_list.append(idx)
        # Clear the square region around the point
        x, y = idx[0], idx[1]
        xmin, ymin = max(0, x - r // 2), max(0, y - r // 2)
        xmax, ymax = min(scores.shape[0], x + r // 2), min(scores.shape[1], y + r // 2)
        scores[ xmin:xmax,ymin:ymax] = -9
    maxvals_list=np.array(maxvals_list,dtype=np.float16)
    preds_list=np.array(preds_list,dtype=np.float16)
    return maxvals_list, preds_list

def visual_mask(image_path, mask, text_left=None, text_right=None, save_path='./tmp.jpg'):
    # Open the image file.
    image = Image.open(image_path).convert("RGBA").resize((1600,1200),resample=Image.Resampling.BILINEAR)

    # Create a blue mask.
    mask_np = np.array(mask)
    mask_blue = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
    mask_blue[..., 2] = 255
    mask_blue[..., 3] = (mask_np * 127.5).astype(np.uint8)

    # Convert mask to an image.
    mask_image = Image.fromarray(mask_blue)

    # Overlay the mask onto the original image.
    composite = Image.alpha_composite(image, mask_image)

    # Draw the texts if provided.
    draw = ImageDraw.Draw(composite)
    font = ImageFont.truetype('arial.ttf', size=30)

    # Draw multi-line text on the left top corner
    if text_left is not None:
        lines = text_left
        y_text = 10
        for line in lines:
            draw.text((10, y_text), line, fill="white", font=font)
            y_text += font.getsize(line)[1]  # Move to the next line position

    # Draw text on the right top corner
    if text_right is not None:
        text_size = draw.textsize(text_right, font=font)
        right_x = composite.width - text_size[0] - 10  # Adjusts for right alignment
        draw.text((right_x, 10), text_right, fill="white", font=font)

    # Convert back to RGB mode (no transparency) and save the image.
    rgb_image = composite.convert("RGB")
    rgb_image.save(save_path)
