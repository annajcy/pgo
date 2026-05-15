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
$mklBinIntel64Dir = Join-Path $mklRoot "bin\intel64"
$mklRedistDir = Join-Path $mklRoot "redist\intel64"
$mklRedistMklDir = Join-Path $mklRoot "redist\intel64\mkl"
$oneApiRoot = Split-Path -Parent (Split-Path -Parent $mklRoot)
$compilerBinDir = Join-Path $oneApiRoot "compiler\latest\bin"

$requiredRuntimeDlls = @(
    "mkl_core.2.dll",
    "mkl_def.2.dll"
)
$optionalRuntimeDlls = @(
    "mkl_rt.2.dll",
    "mkl_intel_lp64.2.dll",
    "mkl_intel_thread.2.dll",
    "libiomp5md.dll"
)
$runtimeSearchRoots = @(
    $mklRoot,
    (Join-Path $oneApiRoot "compiler\latest")
) | Where-Object { Test-Path $_ }
$discoveredRuntimeDirs = @()
foreach ($runtimeDll in $requiredRuntimeDlls) {
    $runtimeFile = $runtimeSearchRoots |
        ForEach-Object { Get-ChildItem -Path $_ -Recurse -Filter $runtimeDll -File -ErrorAction SilentlyContinue } |
        Select-Object -First 1
    if (-not $runtimeFile) {
        throw "Intel oneMKL runtime DLL was not found: $runtimeDll"
    }
    $discoveredRuntimeDirs += $runtimeFile.DirectoryName
}
foreach ($runtimeDll in $optionalRuntimeDlls) {
    $runtimeFile = $runtimeSearchRoots |
        ForEach-Object { Get-ChildItem -Path $_ -Recurse -Filter $runtimeDll -File -ErrorAction SilentlyContinue } |
        Select-Object -First 1
    if ($runtimeFile) {
        $discoveredRuntimeDirs += $runtimeFile.DirectoryName
    } else {
        Write-Warning "Optional Intel oneMKL runtime DLL was not found: $runtimeDll"
    }
}

$runtimeDirs = @(
    @(
        $mklBinDir,
        $mklBinIntel64Dir,
        $mklRedistDir,
        $mklRedistMklDir,
        $mklLibDir,
        $compilerBinDir,
        $discoveredRuntimeDirs
    ) | Where-Object { Test-Path $_ } | Select-Object -Unique
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
