"""
NVML compatibility fix for PyTorch 2.10+ with old NVIDIA drivers.

This module provides utilities to detect and fix the NVML symbol issue
that occurs when using PyTorch 2.10+ with NVIDIA drivers < 470.

The issue: PyTorch 2.10 calls nvmlDeviceGetNvLinkRemoteDeviceType which
doesn't exist in older drivers, causing an AttributeError.

The fix: Inject a stub symbol using LD_PRELOAD or LD_LIBRARY_PATH.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

# The missing symbol that causes the issue
MISSING_SYMBOL = "nvmlDeviceGetNvLinkRemoteDeviceType"


def check_nvml_compatibility() -> bool:
    """
    Check if the current environment has NVML compatibility issues.

    Returns:
        True if NVML issue is detected (fix needed), False otherwise.
    """
    try:
        import torch

        # Check PyTorch version
        torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if torch_version < (2, 4):
            logger.info(f"PyTorch {torch.__version__} - NVML issue not expected (version < 2.4)")
            return False

        # Try to import CUDA extension which triggers NVML loading
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping NVML check")
            return False

        # Try to access nvml functions
        try:
            import torch.cuda.nvtx

            # More direct check: try to use torch.cuda.get_device_properties
            # which internally uses NVML
            _ = torch.cuda.get_device_name(0)
            logger.info("NVML appears to be working correctly")
            return False
        except AttributeError as e:
            if MISSING_SYMBOL in str(e):
                logger.warning(f"NVML compatibility issue detected: {e}")
                return True
            raise
    except Exception as e:
        logger.debug(f"Could not check NVML compatibility: {e}")
        return False


def get_nvml_fix_dir() -> str:
    """
    Get the directory where NVML fix files should be stored.

    Returns:
        Path to NVML fix directory.
    """
    # Use /tmp/nvml_fix as default location
    return "/tmp/nvml_fix"


def apply_nvml_fix_env() -> dict[str, str]:
    """
    Apply NVML fix by modifying environment variables.

    Returns:
        Dict of environment variables that should be set.

    Note:
        This function only returns the environment variables;
        the caller must actually set them in os.environ.
    """
    fix_dir = get_nvml_fix_dir()

    # Check if fix directory exists
    if not os.path.exists(fix_dir):
        logger.warning(
            f"NVML fix directory not found: {fix_dir}. "
            f"Run 'bash scripts/setup_nvml_fix.sh' first."
        )
        return {}

    # Add fix directory to LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if fix_dir not in ld_library_path:
        new_ld_path = f"{fix_dir}:{ld_library_path}" if ld_library_path else fix_dir
        return {"LD_LIBRARY_PATH": new_ld_path}

    return {}


def setup_nvml_fix() -> bool:
    """
    Automatically setup NVML fix if needed.

    This function checks for NVML compatibility issues and attempts
    to apply the fix by running the setup script.

    Returns:
        True if fix was applied or not needed, False if fix failed.
    """
    if not check_nvml_compatibility():
        return True

    logger.info("NVML compatibility issue detected, attempting to apply fix...")

    # Try to run the setup script
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "scripts",
        "setup_nvml_fix.sh"
    )

    if not os.path.exists(script_path):
        logger.error(f"NVML setup script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("NVML fix setup completed successfully")
            logger.info("Please restart Python with: LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH")
            return True
        else:
            logger.error(f"NVML fix setup failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to run NVML setup script: {e}")
        return False


def verify_nvml_fix() -> bool:
    """
    Verify that the NVML fix is working.

    Returns:
        True if fix is working (or not needed), False otherwise.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("CUDA not available, NVML fix verification skipped")
            return True

        # Try to access GPU properties
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Successfully accessed GPU: {device_name}")
        return True
    except AttributeError as e:
        if MISSING_SYMBOL in str(e):
            logger.error(f"NVML fix not working: {e}")
            return False
        raise
    except Exception as e:
        logger.error(f"NVML fix verification failed: {e}")
        return False


def print_nvml_fix_instructions() -> None:
    """Print instructions for applying the NVML fix manually."""
    print("\n" + "=" * 60)
    print("NVML Compatibility Fix Instructions")
    print("=" * 60)
    print("\nThe following command is required to run PyTorch 2.10+ on this system:")
    print("\n  LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH <your-command>")
    print("\nFor example:")
    print("\n  LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH pytest tests/ -v")
    print("\nTo make this persistent, add to your ~/.bashrc or ~/.zshrc:")
    print("\n  export LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH")
    print("\nFirst time setup:")
    print("\n  bash scripts/setup_nvml_fix.sh")
    print("\n" + "=" * 60 + "\n")


def auto_apply_nvml_fix() -> None:
    """
    Automatically apply NVML fix by modifying os.environ.

    This should be called early in the script before importing torch.
    """
    fix_dir = get_nvml_fix_dir()

    # Check if fix exists
    if not os.path.exists(fix_dir):
        logger.debug(f"NVML fix directory not found: {fix_dir}")
        return

    # Modify LD_LIBRARY_PATH in current environment
    current_path = os.environ.get("LD_LIBRARY_PATH", "")
    if fix_dir not in current_path:
        new_path = f"{fix_dir}:{current_path}" if current_path else fix_dir
        os.environ["LD_LIBRARY_PATH"] = new_path
        logger.info(f"Applied NVML fix: LD_LIBRARY_PATH={new_path[:100]}...")
