#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Setting up Python Dev Container environment..."

# 1. Create the .devcontainer directory
mkdir -p .devcontainer

# 2. Create the Dockerfile
cat << 'EOF' > .devcontainer/Dockerfile
# Use the official Microsoft Python Dev Container image
FROM mcr.microsoft.com/devcontainers/python:3.11

# Create the virtual environment OUTSIDE the workspace folder to prevent local sync issues
RUN python3 -m venv /opt/venv

# Give the default non-root user (vscode) ownership of the venv
RUN chown -R vscode:vscode /opt/venv

# Activate the virtual environment permanently by prepending it to the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to the latest version inside the venv
RUN pip install --upgrade pip
EOF
echo "✅ Created .devcontainer/Dockerfile"

# 3. Create the devcontainer.json
cat << 'EOF' > .devcontainer/devcontainer.json
{
  "name": "Clean Python Environment",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/opt/venv/bin/python",
        "terminal.integrated.defaultProfile.linux": "bash"
      },
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  },
  "postCreateCommand": "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi",
  "remoteUser": "vscode"
}
EOF
echo "✅ Created .devcontainer/devcontainer.json"

# 4. Set up .gitignore
GITIGNORE_ENTRIES=".venv/\nvenv/\n__pycache__/\n*.pyc"
if [ ! -f .gitignore ]; then
    echo -e "$GITIGNORE_ENTRIES" > .gitignore
    echo "✅ Created .gitignore"
else
    # Append if not already there to avoid duplicates
    if ! grep -q ".venv/" .gitignore; then
        echo -e "\n$GITIGNORE_ENTRIES" >> .gitignore
        echo "✅ Updated existing .gitignore"
    else
        echo "⚡ .gitignore already contains venv rules."
    fi
fi

# 5. Create requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    touch requirements.txt
    echo "✅ Created empty requirements.txt"
fi

echo "🎉 Dev Container setup complete!"
echo "👉 Next step: Open this folder in VS Code, press Cmd+Shift+P, and select 'Dev Containers: Reopen in Container'."