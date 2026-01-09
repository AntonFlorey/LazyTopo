#!/bin/bash

# Path to blender executable (adjust if not in PATH)
BLENDER_PATH="blender"

# All paths
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_PATH="$BASE_PATH/lazytopo"
LEAN_ADDON_PATH="$BASE_PATH/lean_addon"
PARENT_PATH="$(dirname "$BASE_PATH")"
MPFP_WHEELS_PATH="$PARENT_PATH/agplib/wheelhouse"

# Filenames
LICENSE="LICENSE"
INITFILE="__init__.py"
MANIFEST="blender_manifest.toml"

# Target path
BUILD_PATH="$BASE_PATH/LazyTopoBuilds"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_PATH"

# If lean addon exists, delete it
if [ -d "$LEAN_ADDON_PATH" ]; then
    echo "Removing old lean addon..."
    rm -rf "$LEAN_ADDON_PATH"
fi

# Copy everything necessary to lean addon folder
mkdir -p "$LEAN_ADDON_PATH"
cp -r "$ADDON_PATH"/* "$LEAN_ADDON_PATH/"

# Remove first 11 lines of init file (the bl_info block)
tail -n +12 "$ADDON_PATH/$INITFILE" > "$LEAN_ADDON_PATH/$INITFILE"

# Add license file
cp "$BASE_PATH/$LICENSE" "$LEAN_ADDON_PATH/"

# Copy wheels
mkdir -p "$LEAN_ADDON_PATH/wheels"
find "$MPFP_WHEELS_PATH" -maxdepth 1 -name "agplib*.whl" -exec cp {} "$LEAN_ADDON_PATH/wheels/" \;

# Automatically add wheel names to manifest
{
    echo "wheels = ["
    for wheel in "$LEAN_ADDON_PATH/wheels"/agplib*.whl; do
        echo "  \"./wheels/$(basename "$wheel")\","
    done
    echo "]"
} >> "$LEAN_ADDON_PATH/$MANIFEST"

# Remove pycache if it exists
if [ -d "$LEAN_ADDON_PATH/__pycache__" ]; then
    echo "Removing pycache..."
    rm -rf "$LEAN_ADDON_PATH/__pycache__"
fi

# Call blender build commands
"$BLENDER_PATH" --command extension build --source-dir "$LEAN_ADDON_PATH" --output-dir "$BUILD_PATH" --split-platforms &
"$BLENDER_PATH" --command extension build --source-dir "$LEAN_ADDON_PATH" --output-dir "$BUILD_PATH" &

# Wait for processes to finish
sleep 20

# Cleanup
if [ -d "$LEAN_ADDON_PATH" ]; then
    echo "Removing lean addon..."
    rm -rf "$LEAN_ADDON_PATH"
fi

echo "Build process complete."
