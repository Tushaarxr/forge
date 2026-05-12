#!/bin/bash
set -e

echo "Installing forge-agent..."

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null || {
  echo "Error: Python 3.10+ required"
  exit 1
}

# Prefer pipx if available, fall back to pip
if command -v pipx &> /dev/null; then
  pipx install forge-coder
else
  echo "pipx not found, installing with python3 -m pip..."
  python3 -m pip install forge-coder
fi

echo ""
echo "forge installed. Run 'forge setup' to get started."
