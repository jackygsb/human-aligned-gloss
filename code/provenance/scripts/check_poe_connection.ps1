$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$projectRoot = Split-Path $PSScriptRoot -Parent
$encrypted = [IO.File]::ReadAllBytes((Join-Path $projectRoot '.local-secrets\poe_api_key.dpapi'))
$bytes = [Security.Cryptography.ProtectedData]::Unprotect($encrypted,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
$apiKey = [Text.Encoding]::UTF8.GetString($bytes)
[Array]::Clear($bytes,0,$bytes.Length)
try {
    $balance = Invoke-RestMethod -Uri 'https://api.poe.com/usage/current_balance' -Headers @{Authorization=('Bearer '+$apiKey)} -Method Get -TimeoutSec 45 -MaximumRedirection 0
    $status = @{configured=$true;api_verified=$true;current_point_balance=$balance.current_point_balance;checked_at=(Get-Date).ToString('o');inference_requests=0}
    $status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'mllm_material\reports\poe_connection_status.json') -Encoding UTF8
    $status | ConvertTo-Json
} catch {
    $statusCode = $null
    if ($_.Exception.Response) { $statusCode = [int]$_.Exception.Response.StatusCode }
    @{api_verified=$false;http_status=$statusCode;error_type=$_.Exception.GetType().Name} | ConvertTo-Json
    exit 1
} finally { $apiKey=$null; $encrypted=$null }
