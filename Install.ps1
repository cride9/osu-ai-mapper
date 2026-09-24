$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } else {
        throw 'Install 64-bit Python 3.12 from python.org, then run Install.ps1 again.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.12 environment.' }
}
& '.\.venv\Scripts\python.exe' -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip install --no-deps -e '.[test]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Installation complete. Open Start.cmd to launch the mapper.'

