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
