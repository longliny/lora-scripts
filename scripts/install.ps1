$Env:HF_HOME = "huggingface"

if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    $managedPython = 'runtime\python\cpython-3.12.15-windows-x86_64-none\python.exe'
    if (Test-Path -LiteralPath $managedPython) {
        & $managedPython -m venv venv
    } else {
        python -m venv venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed. Use 64-bit Python 3.12.15.' }
}
.\venv\Scripts\activate

Write-Output "Installing deps..."

python -m pip uninstall -y xformers
python -m pip install -r requirements-cu130.txt
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
python -m pip install --upgrade -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
python -m pip install --no-deps -r requirements-flash-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'Flash Attention installation failed' }
python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed' }

Write-Output "Install completed"
Read-Host | Out-Null ;
