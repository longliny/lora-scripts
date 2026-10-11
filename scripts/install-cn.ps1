$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
function InstallFail {
    Write-Output "安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}
if (Test-Path -Path "python\python.exe") {
    Write-Output "使用 python 文件夹内的 python..."
    $py_path = (Get-Item "python").FullName
    $env:PATH = "$py_path;$env:PATH"
}
else {
    if (!(Test-Path -Path "venv")) {
        Write-Output "正在创建虚拟环境..."
        $managedPython = 'runtime\python\cpython-3.12.15-windows-x86_64-none\python.exe'
        if (Test-Path -LiteralPath $managedPython) {
            & $managedPython -m venv venv
        } else {
            python -m venv venv
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Output "创建虚拟环境失败，请安装 64 位 Python 3.12.15，并检查 PATH 或项目 runtime/python 目录。"
            InstallFail
        }
    }
    
    Write-Output "检测到虚拟环境，尝试激活..."
    .\venv\Scripts\activate
    Check "激活虚拟环境失败。"
}

Write-Output "安装程序所需依赖 (已进行国内加速，若在国外或无法使用加速源请换用 install.ps1 脚本)"
Write-Output "受限于国内加速镜像，torch 安装无法使用镜像源，安装较为缓慢。"
$install_torch = Read-Host "是否需要安装 PyTorch（CUDA 13.0）? [y/n] (默认为 y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq "") {
    python -m pip uninstall -y xformers
    python -m pip install -r requirements-cu130.txt
    Check "torch 安装失败，请删除 venv 文件夹后重新运行。"
}

python -m pip install --upgrade -r requirements.txt
Check "训练界面依赖安装失败。"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq "") {
    python -m pip install --no-deps -r requirements-flash-windows.txt
    Check "Flash Attention 安装失败。"
}
python -m pip check
Check "依赖版本检查失败。"

Write-Output "安装完毕"
Read-Host | Out-Null ;
