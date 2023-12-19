from PIL import Image,ImageFont,ImageDraw
import numpy as np
def visual_mask(image_path, mask,text=None,save_path='./tmp.jpg'):
    # Open the image file.
    image = Image.open(image_path).convert("RGBA").resize((800,600),resample=Image.Resampling.BILINEAR)  # Convert image to RGBA
    # Create a blue mask.
    mask_np = np.array(mask)
    mask_blue = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)  # 4 for RGBA
    mask_blue[..., 2] = 255  # Set blue channel to maximum
    mask_blue[..., 3] = (mask_np * 127.5).astype(np.uint8)  # Adjust alpha channel according to the mask value

    # Convert mask to an image.
    mask_image = Image.fromarray(mask_blue)

    # Overlay the mask onto the original image.
    composite = Image.alpha_composite(image, mask_image)
    # Define font and size.
    if text is not None:
        draw = ImageDraw.Draw(composite)
        font = ImageFont.truetype( 'arial.ttf',size=30)  # 20 is the font size. Adjust as needed.

        draw.text((10, 10), text, fill="white", font=font)  # Prints the text in the top-left corner with a 
        # Convert back to RGB mode (no transparency).
    rgb_image = composite.convert("RGB")
    # Save the image with mask to the specified path.
    rgb_image.save(save_path)
    
def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

