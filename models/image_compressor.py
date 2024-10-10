"""Used to compress high-resolution PNG image to non-spotable compresssed JPG image"""

from PIL import Image
import os


def compress_image(origin, target, quality=90):
    """Load in PNG image and compress it to JPG image"""
    im = Image.open(origin).convert("RGBA")
    x, y = im.size
    bg = Image.new("RGBA", (x, y), (255, 255, 255))
    bg.paste(im, (0, 0, x, y), im)
    bg = bg.convert("RGB")
    bg.save(target, quality=quality)


# Compress PNG texture of the entire library
# Loop for subfolders
object_folder = "ycb"

curr_dir = os.getcwd() + "/" + object_folder + "/"
for subdir in os.listdir(curr_dir):
    curr_path = os.path.join(curr_dir, subdir)
    if not os.path.isdir(curr_path):
        continue

    # one more level for YCB objects
    if object_folder == "ycb":
        curr_path = curr_path + "/google_16k/"

    file_name = curr_path + "/texture_map.png"
    target = curr_path + "/texture_map.jpg"
    if not os.path.exists(file_name):
        continue

    print("Compressing", subdir)
    compress_image(file_name, target, quality=80)
