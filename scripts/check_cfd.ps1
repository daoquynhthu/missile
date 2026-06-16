param(
    [string]$Config = "Release",
    [string]$BuildDir = "build",
    [switch]$FullBuild
)

$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
$BuildPath = Join-Path $Root $BuildDir
$LogDir = Join-Path $BuildPath "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$NoisePatterns = @(
    "^\-\- Could NOT find ",
    "^\-\- Found unsuitable Qt",
    "^\-\- Checking for one of the modules",
    "^\-\- A version of Pastix",
    "^\-\- Performing Test COMPILER_SUPPORT_.* - Failed",
    "^\-\- \*+",
    "^\-\- Build type:",
    "^\-\- Build site:",
    "^\-\- Build string:",
    "^\-\- Enabled backends:",
    "^\-\- Disabled backends:",
    "^\-\- Default order:",
    "^\-\- Maximal matrix",
    "^\-\- SSE",
    "^\-\- AVX",
    "^\-\- Altivec",
    "^\-\- VSX",
    "^\-\- MIPS",
    "^\-\- ARM",
    "^\-\- S390X",
    "^\-\- C\+\+11:",
    "^\-\- SYCL:",
    "^\-\- CUDA:",
    "^\-\- HIP:",
    "^ CXX:",
    "^ CXX_FLAGS:",
    "^ Sparse lib flags:",
    "^\-\- Available targets",
    "^\-\- \-\-\-\-\-\-\-\-\-",
    "^\-\- Target\s+\|",
    "^skip snippet ",
    "^CMake Warning \(dev\) at .*[\\/]_deps[\\/]eigen-src",
    "^CMake Deprecation Warning at .*_deps[\\/]eigen-src",
    "^Call Stack \(most recent call first\):$",
    "^\s+build/_deps/eigen-src/",
    "^\s+CMakeLists.txt:29 \(FetchContent_Declare\)"
)

$SignalPatterns = @(
    "fatal error",
    "\berror\b",
    "\bFAILED\b",
    "\*\*\*Failed",
    "FAIL:",
    "CMake Error",
    "nvcc fatal",
    "100% tests passed",
    "\d+ /\s*\d+ tests PASSED",
    "Test project ",
    "The following tests FAILED"
)

function Test-MatchesAny($Line, $Patterns) {
    foreach ($pattern in $Patterns) {
        if ($Line -match $pattern) { return $true }
    }
    return $false
}

function Get-FilteredLines($Lines) {
    $filtered = New-Object System.Collections.Generic.List[string]
    foreach ($line in $Lines) {
        if (Test-MatchesAny $line $NoisePatterns) { continue }
        if (Test-MatchesAny $line $SignalPatterns) {
            $filtered.Add($line)
        }
    }
    return $filtered
}

function Get-FailureContext($Lines) {
    $context = New-Object System.Collections.Generic.List[string]
    foreach ($line in $Lines) {
        if (-not (Test-MatchesAny $line $NoisePatterns)) {
            $context.Add($line)
        }
    }
    if ($context.Count -gt 80) {
        return $context | Select-Object -Last 80
    }
    return $context
}

function Invoke-LoggedStep($Name, $LogName, $FilePath, [string[]]$Arguments) {
    $LogPath = Join-Path $LogDir $LogName
    $StdoutPath = "$LogPath.stdout"
    $StderrPath = "$LogPath.stderr"
    Write-Host "[check] $Name"
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
    $exitCode = $process.ExitCode
    $output = @()
    if (Test-Path $StdoutPath) { $output += Get-Content $StdoutPath }
    if (Test-Path $StderrPath) { $output += Get-Content $StderrPath }
    $output | Set-Content -Path $LogPath
    Remove-Item -LiteralPath $StdoutPath, $StderrPath -ErrorAction SilentlyContinue

    $signals = Get-FilteredLines $output
    foreach ($line in $signals) {
        Write-Host $line
    }

    if ($exitCode -ne 0) {
        Write-Host "[check] $Name failed, log: $LogPath"
        foreach ($line in (Get-FailureContext $output)) {
            Write-Host $line
        }
        exit $exitCode
    }

    Write-Host "[check] $Name passed, log: $LogPath"
}

Invoke-LoggedStep "configure" "configure.log" "cmake" @("-B", $BuildPath)

if ($FullBuild) {
    Invoke-LoggedStep "build all" "build_all_$Config.log" "cmake" @("--build", $BuildPath, "--config", $Config)
} else {
    $targets = @(
        "TestCfdMesh",
        "TestCfdEuler",
        "TestCfdDiagnostics",
        "TestCfdReconstruction",
        "TestCfdViscous",
        "TestCfdGpu"
    )
    foreach ($target in $targets) {
        Invoke-LoggedStep "build $target" "build_${target}_$Config.log" "cmake" @("--build", $BuildPath, "--target", $target, "--config", $Config)
    }
}

Invoke-LoggedStep "ctest CFD" "ctest_cfd_$Config.log" "ctest" @("--test-dir", $BuildPath, "-C", $Config, "-R", "Cfd(Mesh|Euler|Diagnostics|Reconstruction|Viscous|Gpu)", "--output-on-failure")
