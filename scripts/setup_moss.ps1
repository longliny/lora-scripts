[CmdletBinding()]
param([string]$Python = '', [string]$Proxy = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $Python) { $Python = Join-Path $projectRoot 'venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Create the project venv with scripts/install.ps1 first, or set -Python to your project environment.' }
if ($Proxy) { $env:HTTPS_PROXY = $Proxy; $env:HTTP_PROXY = $Proxy }
& $Python -m pip install -r requirements-cu130.txt
if ($LASTEXITCODE -ne 0) { throw 'Project PyTorch installation failed.' }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Project dependencies installation failed.' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Project dependency validation failed.' }
& $Python -c "import sys; sys.path.insert(0, 'scripts/dev'); import moss_image_train_network as m; import torch; print('Local MOSS sd-scripts ready:', m.__file__, sys.executable, torch.__version__, 'CUDA:', torch.cuda.is_available(), 'arguments:', len(m.setup_parser()._actions))"
if ($LASTEXITCODE -ne 0) { throw 'MOSS runtime import check failed.' }
