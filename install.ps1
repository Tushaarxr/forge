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
    pipx install forge-coder
} else {
    Write-Host "pipx not found, installing with python -m pip..." -ForegroundColor Yellow
    python -m pip install forge-coder
    
    # Calculate pip's script directory
    $pythonPath = python -c "import sys; import os; print(os.path.join(sys.prefix, 'Scripts') if os.name == 'nt' else os.path.join(os.path.expanduser('~'), '.local', 'bin'))"
    
    # Add to current session PATH so they can run it immediately
    if ($env:PATH -notlike "*$pythonPath*") {
        $env:PATH = "$pythonPath;$env:PATH"
        Write-Host "NOTE: Added $pythonPath to current session PATH." -ForegroundColor Gray
        
        # Permanently add to User PATH
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$pythonPath*") {
            $newUserPath = "$pythonPath;$userPath"
            [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
            Write-Host "SUCCESS: Permanently added $pythonPath to your System PATH!" -ForegroundColor Green
            Write-Host "You will be able to run 'forge' in all future terminal windows." -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "forge installed. Run 'forge setup' to get started." -ForegroundColor Green
