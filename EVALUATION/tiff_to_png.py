from PIL import Image
import glob
import os

input_dir = "/Users/mahsa/Downloads/DRIVE/training/images"
output_dir = "Retinal_DRIVE_pngs"
os.makedirs(output_dir, exist_ok=True)



for tiff_file in glob.glob(f"{input_dir}/*.tif*"):
    img = Image.open(tiff_file)

    base = os.path.splitext(os.path.basename(tiff_file))[0]
    out_path = os.path.join(output_dir, base + ".png")

    img.save(out_path, format="PNG")
    print("Converted:", out_path)
