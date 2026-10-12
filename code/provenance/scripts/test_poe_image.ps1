param([ValidateSet('claude-sonnet-4.6','gpt-5.4','gemini-3.1-pro')][string]$Model='claude-sonnet-4.6')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$projectRoot = Split-Path $PSScriptRoot -Parent
$outputFolder = Join-Path $projectRoot 'mllm_material\responses\poe_connection_pilot'
[IO.Directory]::CreateDirectory($outputFolder) | Out-Null
$resultName = if ($Model -eq 'claude-sonnet-4.6') {'claude_sonnet46_single_image.json'} else {$Model+'_single_image.json'}
$resultFile = Join-Path $outputFolder $resultName
if (Test-Path $resultFile) { Write-Output 'Existing pilot record preserved; no duplicate request sent.'; exit }
$encrypted = [IO.File]::ReadAllBytes((Join-Path $projectRoot '.local-secrets\poe_api_key.dpapi'))
$bytes = [Security.Cryptography.ProtectedData]::Unprotect($encrypted,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
$apiKey = [Text.Encoding]::UTF8.GetString($bytes)
[Array]::Clear($bytes,0,$bytes.Length)
try {
    $headers = @{Authorization=('Bearer '+$apiKey)}
    $balance = Invoke-RestMethod -Uri 'https://api.poe.com/usage/current_balance' -Headers $headers -TimeoutSec 30 -MaximumRedirection 0
    if ($balance.current_point_balance -lt 1500) { throw 'Insufficient pilot reserve; no inference sent.' }
    $prompts = Get-Content (Join-Path $projectRoot 'mllm_material\protocol\prompts.json') -Raw | ConvertFrom-Json
    $imagePath = Join-Path $projectRoot 'images_processed\512x512\gloss\img1.png'
    $imageUrl = 'data:image/png;base64,'+[Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))
    $settings = @{max_tokens=256;web_search=$false}
    if ($Model -eq 'claude-sonnet-4.6') {$settings.output_effort='none'}
    if ($Model -eq 'gpt-5.4') {$settings.reasoning_effort='none';$settings.verbosity='low'}
    if ($Model -eq 'gemini-3.1-pro') {$settings.thinking_level='low';$settings.max_tokens=1024}
    $payload = @{model=$Model;stream=$false;messages=@(@{role='system';content=$prompts.system},@{role='user';content=@(@{type='image_url';image_url=@{url=$imageUrl}},@{type='text';text=$prompts.T1})})}
    foreach ($setting in $settings.Keys) {$payload[$setting]=$settings[$setting]}
    $record = @{scope='One-request image API connectivity pilot, not research evidence';model=$Model;image_id='morimoto_0001';prompt=$prompts.T1;image_sha256=(Get-FileHash $imagePath -Algorithm SHA256).Hash;balance_before=$balance.current_point_balance;started_at=(Get-Date).ToString('o');settings=$settings}
    $record | ConvertTo-Json -Depth 12 | Set-Content $resultFile -Encoding UTF8
    $response = Invoke-RestMethod -Uri 'https://api.poe.com/v1/chat/completions' -Headers $headers -Method Post -ContentType 'application/json' -Body ($payload|ConvertTo-Json -Depth 12 -Compress) -TimeoutSec 90 -MaximumRedirection 0
    $record.response=$response
    $record.completed_at=(Get-Date).ToString('o')
    $record | ConvertTo-Json -Depth 20 | Set-Content $resultFile -Encoding UTF8
    $after = Invoke-RestMethod -Uri 'https://api.poe.com/usage/current_balance' -Headers $headers -TimeoutSec 30 -MaximumRedirection 0
    $record.balance_after=$after.current_point_balance
    $record.observed_balance_delta=$record.balance_before-$record.balance_after
    $record | ConvertTo-Json -Depth 20 | Set-Content $resultFile -Encoding UTF8
    @{model=$record.model;answer=$response.choices[0].message.content;finish_reason=$response.choices[0].finish_reason;balance_before=$record.balance_before;balance_after=$record.balance_after;observed_balance_delta=$record.observed_balance_delta} | ConvertTo-Json -Depth 4
} catch {
    $code=$null
    if ($_.Exception.Response) {$code=[int]$_.Exception.Response.StatusCode}
    @{http_status=$code;error_type=$_.Exception.GetType().Name;note='No automatic retry to avoid duplicate charges.'} | ConvertTo-Json
    exit 1
} finally {$apiKey=$null;$headers=$null}
