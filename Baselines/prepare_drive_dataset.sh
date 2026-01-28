#!/bin/bash
# Prepare DRIVE dataset into standard structure with train and test sets.
#
# Input (original DRIVE layout, untouched):
#   Dataset/DRIVE/
#     training/images/*.tif
#     training/1st_manual/*_manual1.gif   (GT for train)
#     training/mask/*_training_mask.gif   (alternative masks)
#     test/images/*_test.tif
#     test/mask/*_test_mask.gif           (GT for test)
#
# Output (new, used by baselines):
#   Dataset/drive/
#     images/*.png        (TRAIN images)
#     masks/*.png         (TRAIN masks)
#   Dataset/drive_test/
#     images/*.png        (TEST images)
#     masks/*.png         (TEST masks)
#
# Filenames are matched by numeric ID (e.g. 21.png, 01.png).

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$ROOT_DIR/Dataset"
SRC_DRIVE="$DATASET_DIR/DRIVE"

TRAIN_OUT="$DATASET_DIR/drive"
TEST_OUT="$DATASET_DIR/drive_test"

mkdir -p "$TRAIN_OUT/images" "$TRAIN_OUT/masks"
mkdir -p "$TEST_OUT/images" "$TEST_OUT/masks"

echo "Preparing DRIVE dataset..."
echo "Source:   $SRC_DRIVE"
echo "Train ->  $TRAIN_OUT"
echo "Test  ->  $TEST_OUT"
echo

convert_to_png() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    sips -s format png "$src" --out "$dst" > /dev/null 2>&1
  else
    echo "WARNING: missing source: $src" >&2
  fi
}

echo "Processing TRAIN split..."
train_img_dir="$SRC_DRIVE/training/images"
train_manual_dir="$SRC_DRIVE/training/1st_manual"
train_mask_dir="$SRC_DRIVE/training/mask"

count_train=0
for img in "$train_img_dir"/*_training.tif; do
  [ -f "$img" ] || continue
  num=$(basename "$img" | sed 's/_training\.tif//')

  # Prefer 1st_manual if available, otherwise fall back to training/mask
  gt="$train_manual_dir/${num}_manual1.gif"
  if [ ! -f "$gt" ]; then
    gt="$train_mask_dir/${num}_training_mask.gif"
  fi

  out_img="$TRAIN_OUT/images/${num}.png"
  out_msk="$TRAIN_OUT/masks/${num}.png"

  convert_to_png "$img" "$out_img"
  convert_to_png "$gt" "$out_msk"

  if [ -f "$out_img" ] && [ -f "$out_msk" ]; then
    ((count_train++))
  else
    echo "WARNING: failed to create pair for ID $num"
  fi
done
echo "  Created $count_train train image-mask pairs."
echo

echo "Processing TEST split..."
test_img_dir="$SRC_DRIVE/test/images"
test_mask_dir="$SRC_DRIVE/test/mask"

count_test=0
for img in "$test_img_dir"/*_test.tif; do
  [ -f "$img" ] || continue
  num=$(basename "$img" | sed 's/_test\.tif//')

  gt="$test_mask_dir/${num}_test_mask.gif"

  out_img="$TEST_OUT/images/${num}.png"
  out_msk="$TEST_OUT/masks/${num}.png"

  convert_to_png "$img" "$out_img"
  convert_to_png "$gt" "$out_msk"

  if [ -f "$out_img" ] && [ -f "$out_msk" ]; then
    ((count_test++))
  else
    echo "WARNING: failed to create TEST pair for ID $num"
  fi
done
echo "  Created $count_test test image-mask pairs."
echo

echo "Verifying 1-to-1 correspondence..."
for split in "$TRAIN_OUT" "$TEST_OUT"; do
  name=$(basename "$split")
  imgs=$(ls -1 "$split/images"/*.png 2>/dev/null | wc -l | tr -d ' ')
  msks=$(ls -1 "$split/masks"/*.png 2>/dev/null | wc -l | tr -d ' ')
  echo "  $name: $imgs images, $msks masks"
done

echo
echo "DRIVE preparation complete."

