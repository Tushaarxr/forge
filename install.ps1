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
    Write-Host "pipx not found, installing with pip..." -ForegroundColor Yellow
    pip install forge-agent
    
    # Check if pip's script directory is in PATH
    $pythonPath = python -c "import sys; import os; print(os.path.join(sys.prefix, 'Scripts') if os.name == 'nt' else os.path.join(os.path.expanduser('~'), '.local', 'bin'))"
    if ($env:PATH -notlike "*$pythonPath*") {
        Write-Host "WARNING: The installation directory ($pythonPath) is NOT in your PATH." -ForegroundColor Yellow
        Write-Host "You may need to restart your terminal or add it manually to run 'forge'." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "forge installed. Run 'forge setup' to get started." -ForegroundColor Green
