param([ValidateSet('chrome','msedge')][string]$Browser='chrome')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskOut=Join-Path $taskRoot 'mllm_material\chatgpt_web_autonomous_v1'
$taskPy=Join-Path $taskRoot 'mllm_runtime\Scripts\python.exe'
$taskScript=Join-Path $PSScriptRoot 'chatgpt_web_runner.py'
$taskProcess=Start-Process -FilePath $taskPy -ArgumentList @('-u',$taskScript,'--browser',$Browser) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskOut 'runner.stdout.log') -RedirectStandardError (Join-Path $taskOut 'runner.stderr.log') -PassThru
$taskProcess.Id | Set-Content (Join-Path $taskOut 'runner.pid')
Write-Output "Runner started: $($taskProcess.Id). Browser: $Browser. Use the separate experiment window."
