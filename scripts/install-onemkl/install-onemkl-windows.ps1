$ErrorActionPreference = "Stop"

$candidateRoots = @(
    "${env:ProgramFiles(x86)}\Intel\oneAPI\mkl\latest",
    "${env:ProgramFiles}\Intel\oneAPI\mkl\latest"
)

$mklRoot = $candidateRoots | Where-Object {
    Test-Path (Join-Path $_ "lib\cmake\mkl\MKLConfig.cmake")
} | Select-Object -First 1

if (-not $mklRoot) {
    winget install --id Intel.oneMKL --exact --silent --accept-package-agreements --accept-source-agreements

    $mklRoot = $candidateRoots | Where-Object {
        Test-Path (Join-Path $_ "lib\cmake\mkl\MKLConfig.cmake")
    } | Select-Object -First 1
}

if (-not $mklRoot) {
    throw "Intel oneMKL installation did not provide lib\cmake\mkl\MKLConfig.cmake"
}

$mklCmakeDir = Join-Path $mklRoot "lib\cmake\mkl"
$mklLibDir = Join-Path $mklRoot "lib\intel64"
$mklBinDir = Join-Path $mklRoot "bin"
$mklRedistDir = Join-Path $mklRoot "redist\intel64"
$oneApiRoot = Split-Path -Parent (Split-Path -Parent $mklRoot)
$compilerBinDir = Join-Path $oneApiRoot "compiler\latest\bin"
$runtimeDirs = @(
    @(
        $mklBinDir,
        $mklRedistDir,
        $mklLibDir,
        $compilerBinDir
    ) | Where-Object { Test-Path $_ }
)

$env:MKLROOT = $mklRoot
$env:MKL_DIR = $mklCmakeDir
$env:CMAKE_PREFIX_PATH = "$mklRoot;$env:CMAKE_PREFIX_PATH"
$env:LIB = "$mklLibDir;$env:LIB"
$env:PATH = ((@($runtimeDirs) + @($env:PATH)) -join ";")

if ($env:GITHUB_ENV) {
    @(
        "MKLROOT=$mklRoot",
        "MKL_DIR=$mklCmakeDir",
        "CMAKE_PREFIX_PATH=$env:CMAKE_PREFIX_PATH",
        "LIB=$env:LIB"
    ) | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
}

if ($env:GITHUB_PATH) {
    $runtimeDirs | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}

Write-Host "oneMKL is installed at $mklRoot"
Write-Host "oneMKL runtime PATH entries:"
$runtimeDirs | ForEach-Object { Write-Host "  $_" }
