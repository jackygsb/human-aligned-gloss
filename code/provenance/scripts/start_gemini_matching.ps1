$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Security
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskEncrypted=[IO.File]::ReadAllBytes((Join-Path $taskRoot '.local-secrets\poe_api_key.dpapi'))
$taskBytes=[Security.Cryptography.ProtectedData]::Unprotect($taskEncrypted,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
try {
 $env:POE_API_KEY=[Text.Encoding]::UTF8.GetString($taskBytes)
 [Array]::Clear($taskBytes,0,$taskBytes.Length)
 & (Join-Path $taskRoot 'mllm_runtime\Scripts\python.exe') (Join-Path $PSScriptRoot 'run_gemini_matching.py')
 if ($LASTEXITCODE -ne 0) {throw 'Gemini pilot stopped; inspect saved status without exposing credentials.'}
 Get-Content (Join-Path $taskRoot 'mllm_material\matching_gemini31pro_v1\status.json')
} finally {Remove-Item Env:POE_API_KEY -ErrorAction SilentlyContinue}
