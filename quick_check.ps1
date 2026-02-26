# Quick build script to check for compilation errors in the library only
$BuildDir = "build"

if (-not (Test-Path $BuildDir)) {
    Write-Host "Build directory not found. Please run build.ps1 first." -ForegroundColor Red
    exit 1
}

cd $BuildDir

# Build only the gravity_lib target (fastest way to check core logic)
Write-Host "Running quick check on gravity_lib..." -ForegroundColor Cyan
cmake --build . --target gravity_lib --config Release --parallel 8

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n--- Quick Check Passed! ---" -ForegroundColor Green
} else {
    Write-Host "`n--- Quick Check Failed! ---" -ForegroundColor Red
    exit 1
}
