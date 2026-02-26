# AeroSim Build Script (Windows/PowerShell)

$BuildDir = "build"

if (-not (Test-Path $BuildDir)) {
    Write-Host "Creating build directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path $BuildDir
}

cd $BuildDir

# Only run CMake configure if necessary (or just run it, it's usually fast)
if (-not (Test-Path "CMakeCache.txt")) {
    Write-Host "Configuring CMake..." -ForegroundColor Yellow
    cmake .. -G "Visual Studio 17 2022" -A x64
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "CMakeCache.txt found, skipping configuration..." -ForegroundColor Cyan
}

# Build the project (incremental build)
Write-Host "Building project..." -ForegroundColor Green
cmake --build . --config Release --parallel 8

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "--- Build Successful ---" -ForegroundColor Green
Write-Host "Output binaries are in: $BuildDir\bin\Release"
cd ..
