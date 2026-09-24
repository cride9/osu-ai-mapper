$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$existingPython = Join-Path $PSScriptRoot '..\..\work\venv\Scripts\python.exe'
if (Test-Path -LiteralPath $projectPython) {
    $mapperPython = (Resolve-Path -LiteralPath $projectPython).Path
    $mapperHome = Join-Path $PSScriptRoot 'local-data'
} elseif (Test-Path -LiteralPath $existingPython) {
    $mapperPython = (Resolve-Path -LiteralPath $existingPython).Path
    $mapperHome = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\work')).Path
} else {
    throw 'Run Install.ps1 first.'
}
if ($env:OSUMAPPER_HOME) { $mapperHome = $env:OSUMAPPER_HOME }
& $mapperPython -m osumapper.cli ui --home $mapperHome

