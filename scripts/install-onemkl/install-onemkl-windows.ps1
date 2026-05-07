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
$mklRedistDir = Join-Path $mklRoot "redist\intel64"

if ($env:GITHUB_ENV) {
    @(
        "MKLROOT=$mklRoot",
        "MKL_DIR=$mklCmakeDir",
        "CMAKE_PREFIX_PATH=$mklRoot;$env:CMAKE_PREFIX_PATH",
        "LIB=$mklLibDir;$env:LIB"
    ) | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
}

if ($env:GITHUB_PATH) {
    @(
        $mklRedistDir,
        $mklLibDir
    ) | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}

Write-Host "oneMKL is installed at $mklRoot"
