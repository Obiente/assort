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
        $Output = Join-Path '.local' ('meetings-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    }
    & cargo run --release --locked -p assort-cli -- train-transcript-demo --output $Output --epochs $Epochs --batch-size $BatchSize --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw 'Transcript training failed. Check the output above.' }

    $checkpoint = Join-Path $Output 'checkpoint'
    $tokenizer = Join-Path $Output 'tokenizer.json'
    & cargo run --release --locked -p assort-cli -- summarize --checkpoint $checkpoint --tokenizer $tokenizer --input 'examples/meeting.json' --output (Join-Path $Output 'example-summary.md')
    if ($LASTEXITCODE -ne 0) { throw 'Saved-model summarization failed.' }
    Get-Content -LiteralPath (Join-Path $Output 'example-summary.md')
    Write-Host "Run and summaries saved to $Output"
} finally {
    Pop-Location
}
