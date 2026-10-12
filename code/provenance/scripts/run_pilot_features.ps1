$ErrorActionPreference = 'Stop'
$env:PATH = 'C:\Users\NCRC-3080TI\Anaconda3\envs\texnet_pytorch;C:\Users\NCRC-3080TI\Anaconda3\envs\texnet_pytorch\Library\bin;C:\Users\NCRC-3080TI\Anaconda3\Library\bin;' + $env:PATH
& 'C:/Users/NCRC-3080TI/Anaconda3/envs/texnet_pytorch/python.exe' "$PSScriptRoot/extract_pilot_features.py" @args
exit $LASTEXITCODE
