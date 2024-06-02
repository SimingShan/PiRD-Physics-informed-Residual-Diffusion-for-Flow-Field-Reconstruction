from PIL import Image
path1 = "output_image/HR reference.png"
path2 = "output_image/y0 LR image.png"
path3 = "output_image/x_0.png"
# Replace these with your actual file paths
image_paths = [path1, path2, path3]
images = [Image.open(image) for image in image_paths]

# Assuming all images have the same height and width
total_width = sum(image.width for image in images)
max_height = max(image.height for image in images)  # Since they are the same, you could also directly use one image's height

# Create a new image with the calculated total width and max height
new_image = Image.new('RGB', (total_width, max_height))

# Paste the images next to each other
x_offset = 0
for image in images:
    new_image.paste(image, (x_offset, 0))
    x_offset += image.width

# Save the new image
new_image.save('combined_image.jpg')
