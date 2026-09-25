from __future__ import annotations

import torch
from packaging.version import Version

# transformers refuses to torch.load .bin weights (the encoder's only published format)
# below 2.6 (CVE-2025-32434). Checked up front so the user gets the fix, not a CVE notice.
MIN_TORCH = "2.6"
TORCH_INSTALL_HINT = "pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126"


def require_supported_torch() -> None:
    if Version(torch.__version__) < Version(MIN_TORCH):
        raise RuntimeError(
            f"PyTorch {torch.__version__} is too old -- version {MIN_TORCH} or newer is required.\n"
            f"Upgrade it (GPU build) with:\n    {TORCH_INSTALL_HINT}"
        )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"GPU ({torch.cuda.get_device_name(device)})"
    return "CPU -- no CUDA GPU detected, this will be very slow (see README 'Troubleshooting')"
