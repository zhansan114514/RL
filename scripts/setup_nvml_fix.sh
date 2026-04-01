#!/bin/bash
# NVML Fix Script for PyTorch 2.10+ with old NVIDIA drivers
#
# Creates a stub library providing nvmlDeviceGetNvLinkRemoteDeviceType,
# then uses patchelf --add-needed to inject it into a copy of libnvidia-ml.so.1.
#
# Usage:
#   bash scripts/setup_nvml_fix.sh
#   # Then run your Python scripts with:
#   # LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH python your_script.py

set -e

NVML_FIX_DIR="/tmp/nvml_fix"
STUB_C="$NVML_FIX_DIR/nvml_stub.c"
STUB_SO="$NVML_FIX_DIR/libnvml_stub.so"
PATCHED_LIB="$NVML_FIX_DIR/libnvidia-ml.so.1"

echo "=== NVML Fix for PyTorch 2.10+ ==="
echo ""

# --- Step 1: Create output directory ---
mkdir -p "$NVML_FIX_DIR"
echo "[1/4] Output directory: $NVML_FIX_DIR"

# --- Step 2: Write and compile stub library ---
echo "[2/4] Compiling stub library with missing symbol..."

cat > "$STUB_C" << 'EOF'
/* Stub for nvmlDeviceGetNvLinkRemoteDeviceType (absent in driver < 470) */
int nvmlDeviceGetNvLinkRemoteDeviceType(void *device, unsigned int link,
                                        int *nvLinkRemoteDeviceType) {
    if (nvLinkRemoteDeviceType) *nvLinkRemoteDeviceType = 0;
    return 0;  /* NVML_SUCCESS */
}
EOF

gcc -shared -fPIC -o "$STUB_SO" "$STUB_C" || {
    echo "Error: Failed to compile stub. Please install gcc."
    exit 1
}
echo "  Stub library: $STUB_SO"

# --- Step 3: Copy libnvidia-ml.so.1 and inject stub via patchelf ---
echo "[3/4] Patching libnvidia-ml.so.1 with patchelf..."

# Locate original libnvidia-ml.so.1
ORIGINAL_LIB=$(ldconfig -p 2>/dev/null | grep 'libnvidia-ml.so.1 ' | head -1 | awk '{print $NF}')
if [ -z "$ORIGINAL_LIB" ] || [ ! -f "$ORIGINAL_LIB" ]; then
    for path in /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
                /usr/lib64/libnvidia-ml.so.1 \
                /usr/lib/libnvidia-ml.so.1; do
        if [ -f "$path" ]; then
            ORIGINAL_LIB="$path"
            break
        fi
    done
fi

if [ -z "$ORIGINAL_LIB" ]; then
    echo "Error: Cannot find libnvidia-ml.so.1 on this system."
    exit 1
fi
echo "  Original library: $ORIGINAL_LIB"

# Copy original into our fix directory
cp "$ORIGINAL_LIB" "$PATCHED_LIB"

# Ensure patchelf is available
if ! command -v patchelf &> /dev/null; then
    echo "  Installing patchelf..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y patchelf
    elif command -v yum &> /dev/null; then
        sudo yum install -y patchelf
    elif command -v conda &> /dev/null; then
        conda install -y -c conda-forge patchelf
    else
        echo "Error: Cannot install patchelf. Install manually:"
        echo "  apt-get install patchelf  OR  conda install -c conda-forge patchelf"
        exit 1
    fi
fi

# Inject stub so that libnvidia-ml.so.1 copy depends on our stub
patchelf --add-needed "$STUB_SO" "$PATCHED_LIB"
echo "  Patched library: $PATCHED_LIB"

# --- Step 4: Print usage instructions ---
echo "[4/4] Done!"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Set LD_LIBRARY_PATH before running your Python scripts:"
echo ""
echo "  export LD_LIBRARY_PATH=$NVML_FIX_DIR:\$LD_LIBRARY_PATH"
echo ""
echo "Or run inline:"
echo ""
echo "  LD_LIBRARY_PATH=$NVML_FIX_DIR:\$LD_LIBRARY_PATH python scripts/03_train_critic.py --config configs/config.yaml"
echo ""
echo "To make it persistent, add the export line to ~/.bashrc or ~/.zshrc."
