"""Download the official public checkpoints and verify their exact SHA-256."""

import hashlib
from pathlib import Path
from urllib.request import urlopen


CHECKPOINTS = {
    "zerodce_Epoch99.pth": (
        "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth",
        "a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612",
    ),
    "zerodcepp_Epoch99.pth": (
        "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth",
        "ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84",
    ),
}


def main():
    project_root = Path(__file__).resolve().parents[2]
    root = project_root / "Results" / "hienanh" / "weights"
    root.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_hash) in CHECKPOINTS.items():
        destination = root / filename
        if destination.exists():
            content = destination.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"Checkpoint checksum mismatch: {filename}")
        if not destination.exists():
            temporary = destination.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(destination)
        print(f"Verified checkpoint: {destination}")


if __name__ == "__main__":
    main()
