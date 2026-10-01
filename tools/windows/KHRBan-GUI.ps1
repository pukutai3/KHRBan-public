[CmdletBinding()]
param([switch]$SelfTest, [string]$LinuxRepoOverride)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$script:Distro = if ($env:KHRBAN_WSL_DISTRO) { $env:KHRBAN_WSL_DISTRO } else { 'Ubuntu-22.04' }
$script:LinuxRepo = if ($LinuxRepoOverride) { $LinuxRepoOverride } else { $env:KHRBAN_LINUX_REPO }
if (-not $script:LinuxRepo -and -not $SelfTest) { throw 'KHRBAN_LINUX_REPO にWSL内のKHRBan絶対パスを設定してください。' }
$script:RuntimeRepo = $script:LinuxRepo
$script:MotionProject = $env:KHRBAN_MOTION_PROJECT
$script:WslExe = Join-Path $env:WINDIR 'System32\wsl.exe'
$script:TrainingProcess = $null
$script:MotionViewerProcess = $null
$script:TensorBoardProcess = $null
$script:WslKeepAliveProcess = $null

$c = @{
    Navy = [Drawing.Color]::FromArgb(18, 42, 58)
    Teal = [Drawing.Color]::FromArgb(0, 126, 132)
    TealDark = [Drawing.Color]::FromArgb(0, 92, 98)
    Orange = [Drawing.Color]::FromArgb(226, 124, 39)
    Red = [Drawing.Color]::FromArgb(183, 55, 55)
    Canvas = [Drawing.Color]::FromArgb(241, 239, 232)
    Panel = [Drawing.Color]::White
    Text = [Drawing.Color]::FromArgb(28, 38, 43)
    Muted = [Drawing.Color]::FromArgb(93, 107, 113)
    Good = [Drawing.Color]::FromArgb(31, 132, 94)
    Warning = [Drawing.Color]::FromArgb(188, 112, 31)
}

function New-Font([float]$Size, [Drawing.FontStyle]$Style = [Drawing.FontStyle]::Regular) {
    New-Object Drawing.Font('Yu Gothic UI', $Size, $Style)
}

function New-Label(
    [string]$Text, [int]$X, [int]$Y, [int]$Width, [int]$Height,
    [float]$FontSize = 9,
    [Drawing.FontStyle]$FontStyle = [Drawing.FontStyle]::Regular,
    [Drawing.Color]$ForeColor = $c.Text
) {
    $cLabel = New-Object Windows.Forms.Label
    $cLabel.Text = $Text
    $cLabel.Location = New-Object Drawing.Point($X, $Y)
    $cLabel.Size = New-Object Drawing.Size($Width, $Height)
    $cLabel.Font = New-Font $FontSize $FontStyle
    $cLabel.ForeColor = $ForeColor
    $cLabel.BackColor = [Drawing.Color]::Transparent
    $cLabel
}

function New-Button(
    [string]$Text, [int]$X, [int]$Y, [int]$Width, [int]$Height,
    [Drawing.Color]$BackColor, [int]$TabIndex, [string]$AccessibleName
) {
    $cButton = New-Object Windows.Forms.Button
    $cButton.Text = $Text
    $cButton.Location = New-Object Drawing.Point($X, $Y)
    $cButton.Size = New-Object Drawing.Size($Width, $Height)
    $cButton.FlatStyle = [Windows.Forms.FlatStyle]::Flat
    $cButton.FlatAppearance.BorderSize = 0
    $cButton.BackColor = $BackColor
    $cButton.ForeColor = [Drawing.Color]::White
    $cButton.Font = New-Font 9 ([Drawing.FontStyle]::Bold)
    $cButton.Cursor = [Windows.Forms.Cursors]::Hand
    $cButton.TabIndex = $TabIndex
    $cButton.AccessibleName = $AccessibleName
    $cButton
}

function Invoke-WslCapture([string[]]$Arguments) {
    if (-not (Test-Path -LiteralPath $script:WslExe)) {
        return [pscustomobject]@{ ExitCode = 127; Output = 'wsl.exe が見つかりません' }
    }
    $output = & $script:WslExe @Arguments 2>&1 | Out-String
    [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $output.Trim() }
}

function Get-ActiveTraining {
    Invoke-WslCapture @(
        '-d', $script:Distro, '--exec', 'pgrep', '-f',
        'khrban-auto-train|python[0-9.]* -m khrban\.(auto_train|auto_train_getup|train)'
    )
}

function Quote-PS([string]$Value) {
    "'" + $Value.Replace("'", "''") + "'"
}

function Start-VisiblePowerShell([string]$Title, [string]$Command) {
    $payload = '$Host.UI.RawUI.WindowTitle = ' + (Quote-PS $Title) + [Environment]::NewLine + $Command
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($payload))
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-NoExit', '-EncodedCommand', $encoded
    ) -WindowStyle Normal -PassThru
}

function Get-WslBaseCommand {
    '& ' + (Quote-PS $script:WslExe) +
        ' -d ' + (Quote-PS $script:Distro) +
        ' --cd ' + (Quote-PS $script:LinuxRepo) + ' --exec'
}

function Start-WslKeepAlive {
    if (
        $script:WslKeepAliveProcess -and
        -not $script:WslKeepAliveProcess.HasExited
    ) {
        return
    }
    $script:WslKeepAliveProcess = Start-Process -FilePath $script:WslExe `
        -ArgumentList @('-d', $script:Distro, '--exec', 'sleep', 'infinity') `
        -WindowStyle Hidden -PassThru
    Start-Sleep -Milliseconds 600
}

function Stop-WslKeepAlive {
    if (-not $script:WslKeepAliveProcess) { return }
    if (-not $script:WslKeepAliveProcess.HasExited) {
        Stop-Process -Id $script:WslKeepAliveProcess.Id -Force `
            -ErrorAction SilentlyContinue
    }
    $script:WslKeepAliveProcess = $null
}

function Add-Activity([string]$Message) {
    $activityBox.AppendText(('[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $Message + [Environment]::NewLine))
    $activityBox.SelectionStart = $activityBox.TextLength
    $activityBox.ScrollToCaret()
}

function Set-Badge([Windows.Forms.Label]$Label, [string]$Text, [string]$State) {
    $Label.Text = $Text
    if ($State -eq 'Good') { $Label.ForeColor = $c.Good }
    elseif ($State -eq 'Warning') { $Label.ForeColor = $c.Warning }
    else { $Label.ForeColor = $c.Muted }
}

function Wait-LocalTcpPort([int]$Port, [int]$TimeoutMs = 20000) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.ElapsedMilliseconds -lt $TimeoutMs) {
        $client = New-Object Net.Sockets.TcpClient
        try {
            $client.Connect('127.0.0.1', $Port)
            return $true
        } catch {
            Start-Sleep -Milliseconds 250
        } finally {
            $client.Dispose()
        }
    }
    return $false
}

if ($SelfTest) {
    $launcherSource = Get-Content -LiteralPath $PSCommandPath -Raw
    $requiredLauncherText = @(
        "New-Button '▶  速度追従歩行'",
        "New-Button '↑  起き上がり開始'",
        ".venv/bin/python -m khrban.auto_train_getup --num-envs 1024",
        "'khrban.auto_train_getup', '--check-velocity-gate'"
    )
    $missingLauncherText = @(
        $requiredLauncherText | Where-Object { -not $launcherSource.Contains($_) }
    )
    if ($missingLauncherText.Count -gt 0) {
        throw ('起き上がり開始ボタンまたは合格ゲート付き起動経路がありません: ' + ($missingLauncherText -join ', '))
    }

    $repo = Invoke-WslCapture @('-d', $script:Distro, '--', 'test', '-d', $script:LinuxRepo)
    $trainer = Invoke-WslCapture @(
        '-d', $script:Distro, '--', 'test', '-x',
        ($script:RuntimeRepo + '/.venv/bin/khrban-auto-train')
    )
    $getupTrainer = Invoke-WslCapture @(
        '-d', $script:Distro, '--', 'test', '-f',
        ($script:LinuxRepo + '/src/khrban/auto_train_getup.py')
    )
    $getupEvaluator = Invoke-WslCapture @(
        '-d', $script:Distro, '--', 'test', '-f',
        ($script:LinuxRepo + '/src/khrban/getup_evaluation.py')
    )
    $getupCli = Invoke-WslCapture @(
        '-d', $script:Distro, '--cd', $script:LinuxRepo, '--exec',
        '.venv/bin/python', '-m', 'khrban.auto_train_getup', '--help'
    )
    $liveViewer = Invoke-WslCapture @(
        '-d', $script:Distro, '--', 'test', '-f',
        ($script:LinuxRepo + '/src/khrban/live_training.py')
    )
    $motionViewer = Invoke-WslCapture @(
        '-d', $script:Distro, '--', 'test', '-x',
        ($script:RuntimeRepo + '/.venv/bin/khrban-motion-viewer')
    )
    $keyboardPolicy = Invoke-WslCapture @(
        '-d', $script:Distro, '--', 'test', '-f',
        ($script:LinuxRepo + '/src/khrban/keyboard_policy.py')
    )
    $motionProject = Invoke-WslCapture @(
        '-d', $script:Distro, '--exec', 'test', '-d', $script:MotionProject
    )
    $trainingProbe = Get-ActiveTraining
    if (
        $repo.ExitCode -ne 0 -or $trainer.ExitCode -ne 0 -or
        $getupTrainer.ExitCode -ne 0 -or $getupEvaluator.ExitCode -ne 0 -or
        $getupCli.ExitCode -ne 0 -or
        $liveViewer.ExitCode -ne 0 -or $motionViewer.ExitCode -ne 0 -or
        $keyboardPolicy.ExitCode -ne 0 -or
        $motionProject.ExitCode -ne 0 -or $trainingProbe.ExitCode -gt 1
    ) {
        throw 'KHRBanの学習またはモーション確認環境を確認できません'
    }
    Write-Output 'KHRBan-GUI self-test: OK (walking + gated get-up buttons / motion / keyboard policy)'
    exit 0
}

$form = New-Object Windows.Forms.Form
$form.Text = 'KHRBan Control Desk'
$form.StartPosition = [Windows.Forms.FormStartPosition]::CenterScreen
$form.ClientSize = New-Object Drawing.Size(700, 820)
$form.MinimumSize = New-Object Drawing.Size(716, 859)
$form.MaximumSize = New-Object Drawing.Size(716, 859)
$form.BackColor = $c.Canvas
$form.Font = New-Font 9
$form.KeyPreview = $true
$form.MaximizeBox = $false
$form.Icon = [Drawing.Icon]::ExtractAssociatedIcon($script:WslExe)

$header = New-Object Windows.Forms.Panel
$header.Location = New-Object Drawing.Point(0, 0)
$header.Size = New-Object Drawing.Size(700, 92)
$header.BackColor = $c.Navy
$form.Controls.Add($header)

$brand = New-Label 'KHRBAN' 24 16 250 38 22 ([Drawing.FontStyle]::Bold) ([Drawing.Color]::White)
$subtitle = New-Label 'KHR-3HV  /  MuJoCo - MJLab reinforcement learning' 27 57 440 22 9 ([Drawing.FontStyle]::Regular) ([Drawing.Color]::FromArgb(191, 213, 220))
$branchChip = New-Label 'BRANCH  確認中' 520 30 150 30 9 ([Drawing.FontStyle]::Bold) ([Drawing.Color]::White)
$branchChip.TextAlign = [Drawing.ContentAlignment]::MiddleCenter
$branchChip.BackColor = $c.TealDark
$header.Controls.AddRange(@($brand, $subtitle, $branchChip))

$statusPanel = New-Object Windows.Forms.Panel
$statusPanel.Location = New-Object Drawing.Point(18, 110)
$statusPanel.Size = New-Object Drawing.Size(664, 112)
$statusPanel.BackColor = $c.Panel
$statusPanel.BorderStyle = [Windows.Forms.BorderStyle]::FixedSingle
$form.Controls.Add($statusPanel)
$statusPanel.Controls.Add((New-Label 'SYSTEM STATUS' 16 11 180 24 10 ([Drawing.FontStyle]::Bold)))
$wslStatus = New-Label '● WSL  確認中' 18 43 195 22 9 ([Drawing.FontStyle]::Bold) $c.Muted
$gpuStatus = New-Label '● GPU  確認中' 226 43 265 22 9 ([Drawing.FontStyle]::Bold) $c.Muted
$trainStatus = New-Label '● 学習  確認中' 18 72 195 22 9 ([Drawing.FontStyle]::Bold) $c.Muted
$motionStatus = New-Label '● モーション  確認中' 226 72 265 22 9 ([Drawing.FontStyle]::Bold) $c.Muted
$refreshButton = New-Button '↻  状態を更新' 510 42 136 42 $c.Teal 1 'WSL、GPU、Git、学習状態を更新'
$statusPanel.Controls.AddRange(@($wslStatus, $gpuStatus, $trainStatus, $motionStatus, $refreshButton))

$trainingPanel = New-Object Windows.Forms.Panel
$trainingPanel.Location = New-Object Drawing.Point(18, 238)
$trainingPanel.Size = New-Object Drawing.Size(664, 248)
$trainingPanel.BackColor = $c.Panel
$trainingPanel.BorderStyle = [Windows.Forms.BorderStyle]::FixedSingle
$form.Controls.Add($trainingPanel)
$trainingPanel.Controls.Add((New-Label 'INDEPENDENT OPERATIONS' 16 11 260 24 10 ([Drawing.FontStyle]::Bold)))
$trainingPanel.Controls.Add((New-Label '学習とモーション確認は別プロセス・別ポートで同時に実行できます' 16 39 620 20 8.5 ([Drawing.FontStyle]::Regular) $c.Muted))
$trainButton = New-Button '▶  速度追従歩行' 16 68 300 48 $c.Orange 2 '自動評価付き速度追従歩行学習を開始'
$getupButton = New-Button '↑  起き上がり開始' 346 68 300 48 $c.Teal 3 '速度追従の評価合格後に起き上がり自動学習を開始'
$stopButton = New-Button '■  学習を停止' 16 128 300 38 $c.Red 4 'この画面から開始した学習だけを停止'
$motionButton = New-Button '◇  モーション確認' 346 128 300 38 $c.Teal 5 'HTH4サンプルモーション確認を開始'
$motionStopButton = New-Button '■  モーション確認を停止' 420 188 226 32 $c.Red 8 'この画面から開始したモーション確認だけを停止'
$stopButton.Enabled = $false
$motionStopButton.Enabled = $false
$testButton = New-Button '全テスト' 16 188 188 32 $c.Navy 6 'KHRBanの全テストを実行'
$smokeButton = New-Button '歩行スモーク' 218 188 188 32 $c.Navy 7 '16環境1反復の歩行PPOスモーク学習を実行'
$configLabel = New-Label '歩行 GPU/8080　起き上がりは歩行合格後　モーション CPU/8081' 16 226 630 16 8 ([Drawing.FontStyle]::Regular) $c.Muted
$configLabel.TextAlign = [Drawing.ContentAlignment]::MiddleCenter
$trainingPanel.Controls.AddRange(@($trainButton, $getupButton, $stopButton, $motionButton, $testButton, $smokeButton, $motionStopButton, $configLabel))

$toolsPanel = New-Object Windows.Forms.Panel
$toolsPanel.Location = New-Object Drawing.Point(18, 502)
$toolsPanel.Size = New-Object Drawing.Size(664, 126)
$toolsPanel.BackColor = $c.Panel
$toolsPanel.BorderStyle = [Windows.Forms.BorderStyle]::FixedSingle
$form.Controls.Add($toolsPanel)
$toolsPanel.Controls.Add((New-Label 'TOOLS' 16 10 120 22 10 ([Drawing.FontStyle]::Bold)))
$terminalButton = New-Button '>_  WSL端末' 16 42 98 36 $c.Navy 9 'KHRBanディレクトリでWSL端末を開く'
$viewerButton = New-Button '◇  実学習ビューア' 122 42 98 36 $c.Navy 10 '実際に学習しているKHRを16体表示'
$keyboardButton = New-Button '⌨ キーボード操作' 228 42 98 36 $c.Teal 11 '最新ポリシーをmicroban互換キーで操作'
$tensorButton = New-Button '▥  TensorBoard' 334 42 98 36 $c.Navy 12 'TensorBoardを起動'
$logsButton = New-Button '□  学習ログ' 440 42 98 36 $c.Navy 13 '学習ログフォルダーを開く'
$githubButton = New-Button '↗  GitHub' 546 42 98 36 $c.Navy 14 'GitHubのtestブランチを開く'
$toolsPanel.Controls.AddRange(@($terminalButton, $viewerButton, $keyboardButton, $tensorButton, $logsButton, $githubButton))
$toolsPanel.Controls.Add((New-Label '各処理は別ウィンドウで実行。終了結果を確認してから次の段階へ進んでください。' 17 88 620 22 8 ([Drawing.FontStyle]::Regular) $c.Muted))

$activityPanel = New-Object Windows.Forms.Panel
$activityPanel.Location = New-Object Drawing.Point(18, 644)
$activityPanel.Size = New-Object Drawing.Size(664, 112)
$activityPanel.BackColor = $c.Panel
$activityPanel.BorderStyle = [Windows.Forms.BorderStyle]::FixedSingle
$form.Controls.Add($activityPanel)
$activityPanel.Controls.Add((New-Label 'ACTIVITY' 16 8 120 21 9 ([Drawing.FontStyle]::Bold)))
$activityBox = New-Object Windows.Forms.TextBox
$activityBox.Location = New-Object Drawing.Point(16, 32)
$activityBox.Size = New-Object Drawing.Size(630, 63)
$activityBox.Multiline = $true
$activityBox.ReadOnly = $true
$activityBox.ScrollBars = [Windows.Forms.ScrollBars]::Vertical
$activityBox.BackColor = [Drawing.Color]::FromArgb(249, 249, 247)
$activityBox.ForeColor = $c.Text
$activityBox.Font = New-Object Drawing.Font('Consolas', 8.5)
$activityPanel.Controls.Add($activityBox)

$repoLabel = New-Label ('WSL: ' + $script:LinuxRepo) 20 778 560 24 8 ([Drawing.FontStyle]::Regular) $c.Muted
$closeButton = New-Button '閉じる' 594 774 88 30 $c.Muted 14 'KHRBan Control Deskを閉じる'
$form.Controls.AddRange(@($repoLabel, $closeButton))

$tip = New-Object Windows.Forms.ToolTip
$tip.SetToolTip($testButton, '.venv/bin/pytest -q')
$tip.SetToolTip($smokeButton, '速度指令を使う16並列環境、1反復の歩行学習確認です')
$tip.SetToolTip($trainButton, 'GPU・8080で速度追従歩行を評価合格まで学習します')
$tip.SetToolTip($getupButton, '速度追従の合格記録とチェックポイントを確認してから、四つの転倒姿勢の起き上がりを自動学習します')
$tip.SetToolTip($stopButton, 'この画面から起動したPowerShellプロセスツリーだけを停止します')
$tip.SetToolTip($motionButton, 'CPU・8081でHTH4サンプルモーション確認を開きます')
$tip.SetToolTip($motionStopButton, 'この画面から起動したモーション確認だけを停止します')
$tip.SetToolTip($viewerButton, 'PPOが実際に更新している16環境を表示します（チェックポイント再生ではありません）')
$tip.SetToolTip($keyboardButton, '最新チェックポイントをCPU・ネイティブMuJoCo 1体で再生します。Vで有効、矢印で移動、Xで停止、Rでリセット、Qで終了')
$tip.SetToolTip($tensorButton, 'http://localhost:6006 を開きます')

$refreshAction = {
    $refreshButton.Enabled = $false
    $refreshButton.Text = '確認中...'
    [Windows.Forms.Application]::DoEvents()

    $probe = Invoke-WslCapture @('-d', $script:Distro, '--', 'test', '-d', $script:LinuxRepo)
    if ($probe.ExitCode -eq 0) { Set-Badge $wslStatus '● WSL  接続済み' 'Good' }
    else { Set-Badge $wslStatus '● WSL  接続不可' 'Warning' }

    $gpu = Invoke-WslCapture @('-d', $script:Distro, '--', 'nvidia-smi', '--query-gpu=name', '--format=csv,noheader')
    if ($gpu.ExitCode -eq 0 -and $gpu.Output) {
        Set-Badge $gpuStatus ('● GPU  ' + (($gpu.Output -split [Environment]::NewLine)[0].Trim())) 'Good'
    } else { Set-Badge $gpuStatus '● GPU  未確認' 'Warning' }

    $git = Invoke-WslCapture @('-d', $script:Distro, '--', 'git', '-C', $script:LinuxRepo, 'status', '--short', '--branch')
    if ($git.ExitCode -eq 0 -and $git.Output) {
        $lines = @($git.Output -split [Environment]::NewLine)
        $branchName = ($lines[0] -replace '^##\s*', '' -replace '\.\..*$', '').Trim()
        $dirty = @($lines | Select-Object -Skip 1 | Where-Object { $_.Trim() })
        if ($dirty.Count -eq 0) {
            $branchChip.Text = 'BRANCH  ' + $branchName
            $branchChip.BackColor = $c.TealDark
        } else {
            $branchChip.Text = '変更あり  ' + $branchName
            $branchChip.BackColor = $c.Warning
        }
    } else {
        $branchChip.Text = 'BRANCH  不明'
        $branchChip.BackColor = $c.Warning
    }

    $training = Get-ActiveTraining
    if ($training.ExitCode -eq 0) {
        Set-Badge $trainStatus '● 学習  実行中' 'Good'
    } else { Set-Badge $trainStatus '● 学習  停止中' 'Muted' }

    $motion = Invoke-WslCapture @('-d', $script:Distro, '--', 'pgrep', '-f', 'khrban[.-]motion[_-]viewer')
    if ($motion.ExitCode -eq 0) {
        Set-Badge $motionStatus '● モーション  実行中' 'Good'
    } else { Set-Badge $motionStatus '● モーション  停止中' 'Muted' }

    if ($script:TrainingProcess -and -not $script:TrainingProcess.HasExited) {
        $stopButton.Enabled = $true
    } else {
        $script:TrainingProcess = $null
        $stopButton.Enabled = $false
    }
    if ($script:MotionViewerProcess -and -not $script:MotionViewerProcess.HasExited) {
        $motionStopButton.Enabled = $true
    } else {
        $script:MotionViewerProcess = $null
        $motionStopButton.Enabled = $false
    }
    $refreshButton.Text = '↻  状態を更新'
    $refreshButton.Enabled = $true
    Add-Activity '状態を更新しました'
}

$refreshButton.Add_Click($refreshAction)
$testButton.Add_Click({
    $base = Get-WslBaseCommand
    [void](Start-VisiblePowerShell 'KHRBan - Tests' ($base + ' .venv/bin/pytest -q'))
    Add-Activity '全テストを別ウィンドウで開始しました'
})
$smokeButton.Add_Click({
    $active = Get-ActiveTraining
    if ($active.ExitCode -eq 0) {
        [Windows.Forms.MessageBox]::Show(
            '本学習の実行中は歩行スモークを開始できません。GPU学習を継続します。',
            'KHRBan 歩行スモーク',
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        Add-Activity '本学習を検出したため、歩行スモークの起動を防止しました'
        return
    }
    $base = Get-WslBaseCommand
    $run = 'gui-smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    [void](Start-VisiblePowerShell 'KHRBan - Walking Smoke Test' ($base + ' .venv/bin/khrban-train --task velocity --num-envs 16 --iterations 1 --run-name ' + $run))
    Add-Activity ('歩行スモークを開始しました: ' + $run)
})
$trainButton.Add_Click({
    $active = Get-ActiveTraining
    if ($active.ExitCode -eq 0) {
        [Windows.Forms.MessageBox]::Show(
            '歩行学習は既に実行中です。重複起動しません。学習ビューアから確認してください。',
            'KHRBan 歩行学習',
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        Add-Activity '既存の歩行学習を検出したため、重複起動を防止しました'
        return
    }
    $nl = [Environment]::NewLine
    $message = '歩行学習を開始します。' + $nl + $nl +
        '1024 envs / 2000反復ごとに自動評価' + $nl +
        '達成まで最新モデルから追加学習（回数上限なし）' + $nl +
        'チェックポイントは最新3件だけ保持' + $nl +
        '保存先: logs/rsl_rl/khr_velocity' + $nl + $nl + '開始しますか？'
    $answer = [Windows.Forms.MessageBox]::Show(
        $message, 'KHRBan 本学習の確認',
        [Windows.Forms.MessageBoxButtons]::YesNo,
        [Windows.Forms.MessageBoxIcon]::Question,
        [Windows.Forms.MessageBoxDefaultButton]::Button2
    )
    if ($answer -ne [Windows.Forms.DialogResult]::Yes) {
        Add-Activity '本学習の開始をキャンセルしました'
        return
    }
    $base = Get-WslBaseCommand
    $run = 'gui-walk-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    $command = $base + ' .venv/bin/khrban-auto-train --num-envs 1536 --iterations-per-round 2000 --run-prefix ' + $run + ' --keep-checkpoints 3 --live-viewer --viewer-num-envs 16'
    $script:TrainingProcess = Start-VisiblePowerShell 'KHRBan - Walking Training' $command
    $stopButton.Enabled = $true
    Add-Activity ('本学習を開始しました: ' + $run)
})
$getupButton.Add_Click({
    $active = Get-ActiveTraining
    if ($active.ExitCode -eq 0) {
        [System.Windows.Forms.MessageBox]::Show(
            '現在のKHR学習または評価が終了してから起き上がり学習を開始してください。',
            'KHRBan 起き上がり学習',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        Add-Activity '実行中のKHR学習を検出したため、起き上がり学習は開始しませんでした'
        return
    }

    $gate = Invoke-WslCapture @(
        '-d', $script:Distro, '--cd', $script:LinuxRepo, '--exec',
        '.venv/bin/python', '-m', 'khrban.auto_train_getup', '--check-velocity-gate'
    )
    if ($gate.ExitCode -ne 0) {
        [System.Windows.Forms.MessageBox]::Show(
            $gate.Output,
            '速度追従はまだ合格していません',
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        Add-Activity '速度追従の合格記録または保持済みチェックポイントがないため、起き上がりを待機します'
        return
    }

    $nl = [Environment]::NewLine
    $message = '速度追従の合格とチェックポイントを確認しました。' + $nl + $nl +
        '起き上がりを1024環境で2000反復ずつ学習し、顔を下・上、左右側面の4姿勢を個別評価します。' + $nl +
        '4姿勢すべてで80%以上が1秒間安定して立つまで追加学習します。' + $nl +
        'チェックポイントは最新3件を保持し、実学習16体を8080番で表示します。' + $nl + $nl +
        '起き上がり学習を開始しますか？'
    $answer = [System.Windows.Forms.MessageBox]::Show(
        $message, 'KHRBan 起き上がり学習の確認',
        [Windows.Forms.MessageBoxButtons]::YesNo,
        [Windows.Forms.MessageBoxIcon]::Question,
        [Windows.Forms.MessageBoxDefaultButton]::Button2
    )
    if ($answer -ne [Windows.Forms.DialogResult]::Yes) {
        Add-Activity '起き上がり学習の開始をキャンセルしました'
        return
    }

    $base = Get-WslBaseCommand
    $run = 'gui-getup-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    $command = $base + ' .venv/bin/python -m khrban.auto_train_getup --num-envs 1024 --iterations-per-round 2000 --run-prefix ' + $run + ' --keep-checkpoints 3 --live-viewer --viewer-num-envs 16 --viewer-port 8080'
    $script:TrainingProcess = Start-VisiblePowerShell 'KHRBan - Get-Up Training' $command
    $stopButton.Enabled = $true
    Add-Activity ('起き上がり自動学習を開始しました: ' + $run)
})
$stopButton.Add_Click({
    if (-not $script:TrainingProcess -or $script:TrainingProcess.HasExited) {
        $stopButton.Enabled = $false
        Add-Activity 'この画面が管理する本学習プロセスはありません'
        return
    }
    $nl = [Environment]::NewLine
    $message = 'この画面から開始した本学習だけを停止します。' + $nl +
        'PID: ' + $script:TrainingProcess.Id + $nl + $nl + '停止しますか？'
    $answer = [Windows.Forms.MessageBox]::Show(
        $message, 'KHRBan 本学習の停止',
        [Windows.Forms.MessageBoxButtons]::YesNo,
        [Windows.Forms.MessageBoxIcon]::Warning,
        [Windows.Forms.MessageBoxDefaultButton]::Button2
    )
    if ($answer -ne [Windows.Forms.DialogResult]::Yes) { return }
    & taskkill.exe /PID $script:TrainingProcess.Id /T /F | Out-Null
    Add-Activity ('本学習を停止しました: PID ' + $script:TrainingProcess.Id)
    $script:TrainingProcess = $null
    $stopButton.Enabled = $false
})
$motionButton.Add_Click({
    if (Wait-LocalTcpPort 8081 500) {
        Start-Process 'http://localhost:8081/'
        Add-Activity '起動済みのモーション確認を開きました（学習は継続）'
        return
    }
    $active = Invoke-WslCapture @('-d', $script:Distro, '--', 'pgrep', '-f', 'khrban[.-]motion[_-]viewer')
    if ($active.ExitCode -eq 0) {
        if (Wait-LocalTcpPort 8081 10000) {
            Start-Process 'http://localhost:8081/'
            Add-Activity '起動中だったモーション確認を開きました'
        } else {
            Add-Activity 'モーション確認プロセスはありますが8081へ接続できません'
        }
        return
    }
    $base = Get-WslBaseCommand
    $command = $base + ' .venv/bin/khrban-motion-viewer ' +
        (Quote-PS $script:MotionProject) + ' --device cpu --port 8081'
    $script:MotionViewerProcess = Start-VisiblePowerShell 'KHRBan - Motion Viewer' $command
    $motionStopButton.Enabled = $true
    Add-Activity 'CPU・8081でモーション確認を起動しています（学習と独立）'
    if (Wait-LocalTcpPort 8081 30000) {
        Start-Process 'http://localhost:8081/'
        Add-Activity 'モーション確認を開きました'
    } else {
        Add-Activity 'モーション確認の起動待ちがタイムアウトしました'
    }
})
$motionStopButton.Add_Click({
    if (-not $script:MotionViewerProcess -or $script:MotionViewerProcess.HasExited) {
        $motionStopButton.Enabled = $false
        Add-Activity 'この画面が管理するモーション確認プロセスはありません'
        return
    }
    & taskkill.exe /PID $script:MotionViewerProcess.Id /T /F | Out-Null
    Add-Activity ('モーション確認だけを停止しました: PID ' + $script:MotionViewerProcess.Id)
    $script:MotionViewerProcess = $null
    $motionStopButton.Enabled = $false
})
$terminalButton.Add_Click({
    $base = Get-WslBaseCommand
    [void](Start-VisiblePowerShell 'KHRBan - WSL Terminal' ($base + ' bash'))
    Add-Activity 'KHRBanディレクトリでWSL端末を開きました'
})
$viewerButton.Add_Click({
    if (Wait-LocalTcpPort 8080 500) {
        Start-Process 'http://localhost:8080/'
        Add-Activity '起動済みの実学習ビューアを開きました'
        return
    }
    [Windows.Forms.MessageBox]::Show(
        '実学習ビューアは学習プロセスと一緒に起動します。「学習を開始」を先に押してください。',
        'KHRBan',
        [Windows.Forms.MessageBoxButtons]::OK,
        [Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
    Add-Activity '実学習ビューアは停止中です。学習開始後に利用できます'
})
$keyboardButton.Add_Click({
    $active = Invoke-WslCapture @(
        '-d', $script:Distro, '--', 'pgrep', '-f',
        'python3? -m khrban\.keyboard_policy|khrban-keyboard-policy'
    )
    if ($active.ExitCode -eq 0) {
        [Windows.Forms.MessageBox]::Show(
            'キーボード操作は既に起動中です。MuJoCoウィンドウを選択してください。',
            'KHRBan キーボード操作',
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        Add-Activity '起動済みのキーボード操作を検出しました'
        return
    }
    $base = Get-WslBaseCommand
    $command = $base + ' .venv/bin/python -m khrban.keyboard_policy --device cpu'
    [void](Start-VisiblePowerShell 'KHRBan - Keyboard Policy' $command)
    Add-Activity '学習と独立したCPU・1体のキーボード操作を起動しました'
})
$tensorButton.Add_Click({
    if ($script:TensorBoardProcess -and -not $script:TensorBoardProcess.HasExited) {
        Start-Process 'http://localhost:6006'
        Add-Activity '起動済みのTensorBoardを開きました'
        return
    }
    $base = Get-WslBaseCommand
    $command = $base + ' .venv/bin/tensorboard --logdir logs/rsl_rl --host 0.0.0.0 --port 6006'
    $script:TensorBoardProcess = Start-VisiblePowerShell 'KHRBan - TensorBoard' $command
    Start-Sleep -Milliseconds 900
    Start-Process 'http://localhost:6006'
    Add-Activity 'TensorBoardを起動し、ブラウザーを開きました'
})
$logsButton.Add_Click({
    $logs = '\\wsl.localhost\' + $script:Distro + ($script:LinuxRepo -replace '/', '\') + '\logs\rsl_rl\khr_velocity'
    if (-not (Test-Path -LiteralPath $logs)) {
        New-Item -ItemType Directory -Path $logs -Force | Out-Null
    }
    Start-Process 'explorer.exe' -ArgumentList $logs
    Add-Activity '学習ログを開きました'
})
$githubButton.Add_Click({
    Start-Process 'https://github.com/pukutai3/KHRBan-public'
    Add-Activity '公開版GitHubを開きました'
})
$closeButton.Add_Click({ $form.Close() })
$form.Add_KeyDown({
    if ($_.KeyCode -eq [Windows.Forms.Keys]::F5) { & $refreshAction }
    elseif ($_.KeyCode -eq [Windows.Forms.Keys]::Escape) { $form.Close() }
})

$timer = New-Object Windows.Forms.Timer
$timer.Interval = 5000
$timer.Add_Tick({
    if ($script:TrainingProcess -and $script:TrainingProcess.HasExited) {
        Add-Activity ('本学習ウィンドウが終了しました: exit ' + $script:TrainingProcess.ExitCode)
        $script:TrainingProcess = $null
        $stopButton.Enabled = $false
    }
    if ($script:MotionViewerProcess -and $script:MotionViewerProcess.HasExited) {
        Add-Activity ('モーション確認ウィンドウが終了しました: exit ' + $script:MotionViewerProcess.ExitCode)
        $script:MotionViewerProcess = $null
        $motionStopButton.Enabled = $false
    }
})
$form.Add_Shown({
    Start-WslKeepAlive
    Add-Activity 'KHRBan Control Deskを起動しました'
    & $refreshAction
    $timer.Start()
})
$form.Add_FormClosed({
    $timer.Stop()
    Stop-WslKeepAlive
})
[void]$form.ShowDialog()
