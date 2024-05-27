from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
import numpy as np 

def visual_sentences(image_path, points, patch_size, label=None, confidences=None, text=None, save_path=None, font_size=60,sample_visual=[]):
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    box_color = 'green' if label == 1 else 'yellow' if label == 2 else 'red'
    # Iterate over each point
    for i, (x, y) in enumerate(points):

        # Set the box color based on the label

        # Calculate the top-left and bottom-right coordinates of the box
        half_size = patch_size // 2
        top_left_x = x - half_size
        top_left_y = y - half_size
        bottom_right_x = x + half_size
        bottom_right_y = y + half_size

        # Draw the box
        draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline=box_color, width=3)

        # Draw the confidence value near the top left of the box
        confidence_text = f"{confidences[i]:.2f}"  # Format confidence to 2 decimal places
        draw.text((top_left_x, top_left_y - font_size - 2), confidence_text, fill=box_color, font=ImageFont.truetype("./arial.ttf", font_size))

    # Draw additional text if provided
    if text is not None:
        # Load the Arial font with the specified font size
        font = ImageFont.truetype("./arial.ttf", font_size)
        text_position = (10, 10)  # Top left corner
        draw.text(text_position, text, fill="white", font=font)
    for x, y in sample_visual:
        # Define the position for the star
        star_position = (x, y)

        # Define the star symbol and its color
        star_symbol = "*"
        star_color = "blue"  # You can choose any color you like

        # Draw the star on the image
        draw.text(star_position, star_symbol, fill=star_color, font=ImageFont.truetype("./arial.ttf", font_size))
    # Save or show the image
    if save_path:
        img.save(save_path)
    else:
        img.show()
def crop_patches(img,patch_size,x,y, max_weight=1599,max_height=1199):
    '''
    keep the size as conv ridge segmentation model
    '''
    left  = int(x - patch_size // 2)
    upper = int(y - patch_size // 2)
    right = int(x + patch_size // 2)
    lower = int(y + patch_size // 2)
    
    # Pad if necessary
    padding = [max(0, -left), max(0, -upper), max(0, right - max_weight), max(0, lower - max_height)]
    left=max(0, left)
    upper=max(0,upper)
    right=min(max_weight, right)
    lower= min(max_height, lower)
    # Crop the patch
    patch = img.crop((left, upper, right, lower))
    patch = ImageOps.expand(patch, tuple(padding), fill=255) 
    
    res = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
    # Create a circular mask for the inscribed circle
    mask = Image.new('L', (patch_size, patch_size), 0)
    draw = ImageDraw.Draw(mask)
    radius = patch_size // 2
    center = (radius, radius)
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)
    res.paste(patch, (0, 0), mask=mask)
    
    return res
    