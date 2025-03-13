#!/bin/bash
set -e

echo "Setting up UV package manager for Ubuntu WSL2"
echo "=============================================="

# Check if UV is already installed
if command -v uv &> /dev/null; then
    echo "UV is already installed."
    uv --version
    echo "To update UV, run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 0
fi

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y curl build-essential libssl-dev libffi-dev python3-dev

# Install UV
echo "Installing UV package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH if it's not already there
UV_PATH="$HOME/.cargo/bin"
if [[ ":$PATH:" != *":$UV_PATH:"* ]]; then
    echo "Adding UV to PATH in your profile..."
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.profile"
    
    # Source the updated profile
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify installation
echo "Verifying UV installation..."
uv --version

# Set up UV environment
echo "Creating a default UV virtual environment..."
mkdir -p "$HOME/.local/share/uv/venvs"

echo "Setting up UV configuration..."
mkdir -p "$HOME/.config/uv"
cat > "$HOME/.config/uv/settings.toml" << EOF
[python]
python-path = "python3"

[venv]
location = "$HOME/.local/share/uv/venvs"

[pypi]
index-url = "https://pypi.org/simple"
EOF

echo ""
echo "UV has been successfully installed!"
echo "To use UV, run commands like:"
echo "  uv venv                         # Create a virtual environment"
echo "  uv pip install <package>        # Install packages"
echo "  uv pip freeze                   # List installed packages"
echo "  uv pip run <command>            # Run a command in an isolated environment"
echo ""
echo "You may need to restart your terminal or run 'source ~/.bashrc' for the PATH changes to take effect."