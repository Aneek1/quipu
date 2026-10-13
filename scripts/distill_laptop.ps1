<#
Distillation v2 on the laptop: llama-server (4-bit teacher) + the generator.
  .\scripts\distill_laptop.ps1 setup   # download llama.cpp + the GGUF (once)
  .\scripts\distill_laptop.ps1 start   # serve + generate parts 5..19 downward
  .\scripts\distill_laptop.ps1 pause   # write PAUSE: generator finishes in-flight answers, then start stops the server
  .\scripts\distill_laptop.ps1 status  # parts, rows, done flags, total rows
Never run alongside a GPU benchmark (they share 8 GB).
Works in Windows PowerShell 5.1.
#>
param([Parameter(Mandatory = $true)][ValidateSet('setup', 'start', 'pause', 'status')][string]$Cmd)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # PS 5.1 Invoke-WebRequest is very slow with the progress bar

$Repo       = Split-Path $PSScriptRoot -Parent
# llama.cpp build b11429 = the tag that release v0.6.0 ("latest") points to; CUDA 12.4 Windows x64 build
$LlamaTag   = 'b11429'
$LlamaCuda  = '12.4'
$GgufRepo   = 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF'
$GgufRev    = '13fb94bfda8c8cf22497dc57b78f391a9acb426a'
$GgufFile   = 'qwen2.5-coder-7b-instruct-q4_k_m.gguf'
$TeacherRev = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'   # Qwen/Qwen2.5-Coder-7B-Instruct
$Tools      = Join-Path $Repo 'tools\llama.cpp'
$Models     = Join-Path $Repo 'models'
$Root       = Join-Path $Repo 'data\distill-v2'
$Py         = Join-Path $Repo '.venv\Scripts\python.exe'
$Port       = 8080
$PidFile    = Join-Path $Root 'server.pid'
# 4 slots x 4096 tokens: a long prompt plus a 1024-token answer fits in one slot
$ServerArgs = '-m "{0}" -ngl 99 -c 16384 -np 4 --host 127.0.0.1 --port {1} --alias teacher' -f (Join-Path $Models $GgufFile), $Port

function Get-LlamaAssets {
    $names = @("llama-$LlamaTag-bin-win-cuda-$LlamaCuda-x64.zip", "cudart-llama-bin-win-cuda-$LlamaCuda-x64.zip")
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    $ghOk = $false
    if ($gh) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'   # PS 5.1 turns redirected native stderr into errors
        try { & gh auth status 2>$null | Out-Null; if ($LASTEXITCODE -eq 0) { $ghOk = $true } } catch { $ghOk = $false }
        $ErrorActionPreference = $prev
    }
    if ($ghOk) {
        foreach ($n in $names) {
            & gh release download $LlamaTag --repo ggml-org/llama.cpp --pattern $n --clobber | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "gh release download failed for $n" }
        }
        return $names
    }
    # No gh: confirm the assets on the GitHub API, then download from github.com/ggml-org/llama.cpp only
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$LlamaTag" -Headers @{ 'User-Agent' = 'quipu-distill' }
    foreach ($n in $names) {
        if (-not ($rel.assets | Where-Object { $_.name -eq $n })) { throw "asset $n not in llama.cpp release $LlamaTag" }
        $url = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/$n"
        Write-Host "downloading $url"   # Write-Output would become part of the return value
        Invoke-WebRequest $url -OutFile (Join-Path $Tools $n) -UseBasicParsing
    }
    return $names
}

function Stop-Server($proc) {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $proc.WaitForExit(15000) | Out-Null
    }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
}

switch ($Cmd) {
    'setup' {
        New-Item -ItemType Directory -Force $Tools, $Models | Out-Null
        if (Test-Path (Join-Path $Tools 'llama-server.exe')) {
            Write-Output "llama.cpp already in $Tools"
        } else {
            Push-Location $Tools
            try {
                $zips = Get-LlamaAssets
                foreach ($z in $zips) { Expand-Archive (Join-Path $Tools $z) -DestinationPath $Tools -Force }
            } finally { Pop-Location }
            if (-not (Test-Path (Join-Path $Tools 'llama-server.exe'))) { throw "llama-server.exe not found in $Tools after unzip" }
        }
        & $Py -c "from huggingface_hub import hf_hub_download as d; print(d('$GgufRepo', '$GgufFile', revision='$GgufRev', local_dir=r'$Models'))"
        if ($LASTEXITCODE -ne 0) { throw 'GGUF download failed' }
    }
    'start' {
        New-Item -ItemType Directory -Force $Root | Out-Null
        # Refuse if a server is already up (a second start, or one orphaned by closing the window)
        if (Test-Path $PidFile) {
            $old = Get-Process -Id ([int](Get-Content $PidFile)) -ErrorAction SilentlyContinue
            if ($old -and $old.ProcessName -eq 'llama-server') {
                throw "llama-server already running (pid $($old.Id)). Run pause and wait, or Stop-Process -Id $($old.Id) if it is orphaned."
            }
            Remove-Item $PidFile
        }
        $busy = $true
        try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5 | Out-Null } catch { $busy = $false }
        if ($busy) { throw "something already answers on port $Port; stop it before start" }
        if (Test-Path (Join-Path $Root 'PAUSE')) {
            Remove-Item (Join-Path $Root 'PAUSE')
            Write-Output 'cleared a pending PAUSE file'
        }
        $server = $null
        $code = 1
        try {
            $server = Start-Process -PassThru -WindowStyle Minimized -FilePath (Join-Path $Tools 'llama-server.exe') -ArgumentList $ServerArgs
            Set-Content -Path $PidFile -Value $server.Id -Encoding Ascii
            $ready = $false
            for ($i = 0; $i -lt 150; $i++) {
                if ($server.HasExited) { throw "llama-server exited during load (code $($server.ExitCode))" }
                try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5 | Out-Null; $ready = $true; break } catch { Start-Sleep 2 }
            }
            if (-not $ready) { throw 'llama-server not healthy after 300 s' }
            & $Py (Join-Path $Repo 'scripts\distill_generate_v2.py') --root 'data/distill-v2' --parts 5:19 --direction down `
                --endpoint "http://127.0.0.1:$Port" --teacher-tag q4km-llamacpp --revision $TeacherRev `
                --concurrency 4 --max-new-tokens 1024
            $code = $LASTEXITCODE
            Write-Output "generator exited with code $code"
        } finally {
            Stop-Server $server
            Write-Output 'llama-server stopped'
        }
        exit $code
    }
    'pause' {
        New-Item -ItemType Directory -Force $Root | Out-Null
        New-Item -ItemType File -Force (Join-Path $Root 'PAUSE') | Out-Null
        Write-Output 'PAUSE file written: the generator stops after in-flight answers; start then stops the server.'
    }
    'status' {
        $total = 0
        $files = Get-ChildItem (Join-Path $Root 'generations') -Filter '*.jsonl' -ErrorAction SilentlyContinue
        foreach ($f in $files) {
            $rows = 0
            # ReadWrite share: the generator holds the file open while it appends
            $fs = New-Object System.IO.FileStream($f.FullName, 'Open', 'Read', 'ReadWrite')
            $sr = New-Object System.IO.StreamReader($fs)
            try {
                while ($null -ne ($line = $sr.ReadLine())) { if ($line.TrimEnd().EndsWith('}')) { $rows++ } }   # skip a half-written tail
            } finally { $sr.Close() }
            $total += $rows
            $done = Test-Path ($f.FullName -replace '\.jsonl$', '.done')
            '{0,-40} rows {1,7}  done {2}' -f $f.Name, $rows, $done
        }
        '{0,-40} rows {1,7}' -f 'TOTAL', $total
        if (Test-Path (Join-Path $Root 'PAUSE')) { 'PAUSE file present' }
        if (Test-Path $PidFile) {
            $id = Get-Content $PidFile
            if (Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue) { 'llama-server running: pid {0}' -f $id } else { 'server.pid is stale (pid {0} gone)' -f $id }
        }
    }
}
