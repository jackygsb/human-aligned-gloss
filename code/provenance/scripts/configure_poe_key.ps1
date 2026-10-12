param([switch]$SelfTest)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$projectRoot = Split-Path $PSScriptRoot -Parent
$secretFolder = Join-Path $projectRoot '.local-secrets'
$secretFile = Join-Path $secretFolder 'poe_api_key.dpapi'
$statusFile = Join-Path $projectRoot 'mllm_material\reports\poe_setup_status.json'
if ($SelfTest) {
    $sample = [Text.Encoding]::UTF8.GetBytes('local-configuration-test-not-a-real-key')
    $sealed = [Security.Cryptography.ProtectedData]::Protect($sample, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
    $opened = [Security.Cryptography.ProtectedData]::Unprotect($sealed, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
    if ([Convert]::ToBase64String($opened) -ne [Convert]::ToBase64String($sample)) { throw 'Encryption verification failed' }
    Write-Output 'Local encryption round-trip passed; no credentials saved or transmitted.'
    exit
}
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[Windows.Forms.Application]::EnableVisualStyles()
$form = New-Object Windows.Forms.Form
$form.Text = 'Poe 实验接口：本机密钥配置'
$form.ClientSize = New-Object Drawing.Size(600,270)
$form.StartPosition = 'CenterScreen'
$form.ShowInTaskbar = $true
$form.TopMost = $true
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.Font = New-Object Drawing.Font('Microsoft YaHei UI',10)
$label = New-Object Windows.Forms.Label
$label.Location = New-Object Drawing.Point(20,18)
$label.Size = New-Object Drawing.Size(560,75)
$label.Text = "请粘贴刚创建的 Poe API 密钥（不是密钥名称）。`r`n保存后由 Windows 为当前用户加密；不进入聊天或实验结果。`r`n本窗口不连接网络、不调用模型，也不消耗积分。"
$form.Controls.Add($label)
$box = New-Object Windows.Forms.TextBox
$box.Location = New-Object Drawing.Point(20,105)
$box.Size = New-Object Drawing.Size(560,30)
$box.UseSystemPasswordChar = $true
$form.Controls.Add($box)
$message = New-Object Windows.Forms.Label
$message.Location = New-Object Drawing.Point(20,150)
$message.Size = New-Object Drawing.Size(560,35)
$form.Controls.Add($message)
$save = New-Object Windows.Forms.Button
$save.Text = '加密保存'
$save.Location = New-Object Drawing.Point(330,205)
$save.Size = New-Object Drawing.Size(120,36)
$save.Add_Click({
    $value = $box.Text.Trim()
    if ($value.Length -lt 16 -or $value -match '\s') { $message.Text = '请粘贴完整密钥；不要填写名称或带空格的文字。'; return }
    try {
        [IO.Directory]::CreateDirectory($secretFolder) | Out-Null
        $bytes = [Text.Encoding]::UTF8.GetBytes($value)
        $encrypted = [Security.Cryptography.ProtectedData]::Protect($bytes,$null,[Security.Cryptography.DataProtectionScope]::CurrentUser)
        [IO.File]::WriteAllBytes($secretFile,$encrypted)
        [Array]::Clear($bytes,0,$bytes.Length)
        $value = $null
        $box.Clear()
        @{configured=$true;stored='Windows DPAPI CurrentUser';api_verified=$false;updated_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
        [Windows.Forms.MessageBox]::Show('已加密保存。尚未发送任何 API 请求。可以回到聊天继续。','配置完成') | Out-Null
        $form.Close()
    } catch { $message.Text = '保存失败，请检查本机目录权限后重试。' }
})
$form.Controls.Add($save)
$cancel = New-Object Windows.Forms.Button
$cancel.Text = '取消'
$cancel.Location = New-Object Drawing.Point(465,205)
$cancel.Size = New-Object Drawing.Size(115,36)
$cancel.Add_Click({$form.Close()})
$form.Controls.Add($cancel)
$form.AcceptButton = $save
$form.CancelButton = $cancel
$form.Add_Shown({$form.BringToFront();$form.Activate();$box.Focus()})
[void]$form.ShowDialog()
$form.Dispose()
