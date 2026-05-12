Write-Host "Installing forge-agent..."

# Check Python version
try {
    python -c "import sys; assert sys.version_info >= (3,10)"
} catch {
    Write-Host "Error: Python 3.10+ required" -ForegroundColor Red
    exit 1
}

# Prefer pipx if available, fall back to pip
if (Get-Command pipx -ErrorAction SilentlyContinue) {
    pipx install forge-agent
} else {
    Write-Host "pipx not found, installing with pip..."
    pip install forge-agent
}

Write-Host ""
Write-Host "forge installed. Run 'forge setup' to get started." -ForegroundColor Green
