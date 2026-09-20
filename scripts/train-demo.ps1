param(
    [int]$Epochs = 20,
    [int]$BatchSize = 16,
    [int]$Seed = 42,
    [string]$Output = ""
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not $Output) {
        $Output = Join-Path '.local' ('support-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    }
    & cargo run --release --locked -p assort-cli -- train-demo --output $Output --epochs $Epochs --batch-size $BatchSize --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw 'Training failed. Check the output above.' }

    & cargo run --release --locked -p assort-cli -- infer --checkpoint (Join-Path $Output 'checkpoint') --tokenizer (Join-Path $Output 'tokenizer.json') --input 'examples/support-requests.json'
    if ($LASTEXITCODE -ne 0) { throw 'Saved-model inference failed.' }
    Write-Host "Run saved to $Output"
} finally {
    Pop-Location
}
