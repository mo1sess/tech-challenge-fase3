param(
    [string]$PythonExecutable = ''
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if ($PythonExecutable) {
    & $PythonExecutable -m venv .venv
}
else {
    try {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python launcher returned an error.' }
    }
    catch {
        throw 'Python 3.12 acessivel nao foi encontrado. Instale-o pelo python.org ou informe -PythonExecutable C:\caminho\python.exe.'
    }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e . -r requirements\dev.txt

Write-Host 'Ambiente local criado. Ative com: .\.venv\Scripts\Activate.ps1'
