param([Parameter(Mandatory=$true)][string]$ScriptName, [switch]$PaperLr)
$ErrorActionPreference = 'Stop'
$env:PATH = 'C:\Users\NCRC-3080TI\Anaconda3\envs\texnet_pytorch;C:\Users\NCRC-3080TI\Anaconda3\envs\texnet_pytorch\Library\bin;C:\Users\NCRC-3080TI\Anaconda3\Library\bin;' + $env:PATH
if ($PaperLr) {
    & 'C:/Users/NCRC-3080TI/Anaconda3/envs/texnet_pytorch/python.exe' "$PSScriptRoot/$ScriptName" --paper-lr
} else {
    & 'C:/Users/NCRC-3080TI/Anaconda3/envs/texnet_pytorch/python.exe' "$PSScriptRoot/$ScriptName" @args
}
exit $LASTEXITCODE
