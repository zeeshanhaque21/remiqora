#!/usr/bin/env bash
# Installs the developer toolchain Remiqora's other scripts need, via
# Homebrew (https://brew.sh). Windows equivalent: setup_prereqs.ps1.
#
# NOT covered here, on purpose:
#   - The NVIDIA CUDA Toolkit / GPU driver step Windows needs - Apple
#     Silicon has no NVIDIA GPU. Generation runs on the built-in Metal
#     backend instead (see setup_models.sh), no separate driver to install.
#   - Xcode Command Line Tools - installing them triggers Apple's own GUI
#     installer, which can't be driven non-interactively; this script only
#     checks for them and tells you to run the installer yourself.
#
# Re-run any time - brew skips formulae that are already installed.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script is for macOS. Use setup_prereqs.ps1 on Windows." >&2
    exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
    echo "Xcode Command Line Tools were not found - triggering the installer..."
    xcode-select --install || true
    echo "Finish that install (a separate window should have opened), then re-run this script."
    exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew was not found. Install it from https://brew.sh, then re-run this script." >&2
    exit 1
fi

echo "=== Toolchain (git, python, uv, node, cmake, ffmpeg, ninja, whisper-cpp) ==="
brew install git python@3.12 uv node cmake ffmpeg ninja whisper-cpp

echo ""
echo "=== Not automated - already built in on Apple Silicon ==="
echo "  GPU acceleration: the Metal backend ships with macOS - nothing to install."

echo ""
echo "Note: cmake/ninja above are only used if you later run"
echo "./setup_models.sh --from-source. By default that script installs YuE2"
echo "from audio.cpp's own prebuilt macOS/Metal release instead - no compiler"
echo "needed at all. --from-source additionally needs full Xcode.app (not just"
echo "the Command Line Tools) for its Metal shader compiler - that part can't"
echo "be scripted (Apple ID in the App Store GUI, then a sudo password), so if"
echo "you ever need it:"
echo "  1. Install Xcode from the App Store: https://apps.apple.com/app/xcode/id497799835"
echo "  2. sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"

echo ""
echo "Done. Close this terminal and open a NEW one (so PATH picks up what just"
echo "installed), then run ./setup_models.sh."
