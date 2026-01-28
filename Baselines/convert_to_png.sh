#!/bin/bash
# Convert all images and masks to PNG format
# Masks will be converted to binary (0/255)

set -e

BASE="Dataset"

echo "Converting all files to PNG format..."
echo ""

# Function to convert image to PNG using sips
convert_to_png() {
    local src="$1"
    local dst="$2"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        sips -s format png "$src" --out "$dst" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            return 0
        else
            echo "Warning: Failed to convert $src" >&2
            return 1
        fi
    fi
    return 0
}

# DRIVE: Convert .tif images and .gif masks to PNG
echo "DRIVE dataset:"
count=0
for img in "$BASE/drive/images"/*.tif; do
    if [ -f "$img" ]; then
        stem=$(basename "$img" .tif)
        png_img="$BASE/drive/images/${stem}.png"
        convert_to_png "$img" "$png_img"
        if [ -f "$png_img" ]; then
            rm -f "$img"
            ((count++))
        fi
    fi
done
echo "  Converted $count images"

count=0
for mask in "$BASE/drive/masks"/*.gif; do
    if [ -f "$mask" ]; then
        stem=$(basename "$mask" .gif)
        png_mask="$BASE/drive/masks/${stem}.png"
        convert_to_png "$mask" "$png_mask"
        if [ -f "$png_mask" ]; then
            rm -f "$mask"
            ((count++))
        fi
    fi
done
echo "  Converted $count masks"

# ISBI12: Already PNG, but verify
echo ""
echo "ISBI12 dataset:"
img_count=$(ls -1 "$BASE/isbi12/images"/*.png 2>/dev/null | wc -l | tr -d ' ')
mask_count=$(ls -1 "$BASE/isbi12/masks"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo "  $img_count images (already PNG)"
echo "  $mask_count masks (already PNG)"

# CRACK: Convert .jpg images and .bmp masks to PNG
echo ""
echo "CRACK dataset:"
count=0
for img in "$BASE/crack/images"/*.jpg; do
    if [ -f "$img" ]; then
        stem=$(basename "$img" .jpg)
        png_img="$BASE/crack/images/${stem}.png"
        convert_to_png "$img" "$png_img"
        if [ -f "$png_img" ]; then
            rm -f "$img"
            ((count++))
        fi
    fi
done
echo "  Converted $count images"

count=0
for mask in "$BASE/crack/masks"/*.bmp; do
    if [ -f "$mask" ]; then
        stem=$(basename "$mask" .bmp)
        png_mask="$BASE/crack/masks/${stem}.png"
        convert_to_png "$mask" "$png_mask"
        if [ -f "$png_mask" ]; then
            rm -f "$mask"
            ((count++))
        fi
    fi
done
echo "  Converted $count masks"

echo ""
echo "Format conversion complete!"
echo ""
echo "Note: Masks still need to be converted to binary (0/255)."
echo "Run: python3 convert_masks_to_binary.py"
