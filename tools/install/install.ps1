<#
.SYNOPSIS
Installs sems from this checkout for the current user and puts it on the PATH.

.DESCRIPTION
For development; users install a release instead (see the README). Builds sems in release mode,
copies it to %LOCALAPPDATA%\sems\bin, and adds that directory to the user PATH. sems downloads ONNX
Runtime and the model itself the first time it needs them.
Open a new terminal afterwards to pick up the PATH change. Running it again updates the install.

.PARAMETER Cuda
Also install the optional CUDA pack (`sems gpu install`, ~1 GB, NVIDIA driver 580+) for fast indexing.

.PARAMETER SkipBuild
Install the already built target\release\sems.exe instead of building it.

.PARAMETER Uninstall
Remove sems.exe and the PATH entry. The model, runtimes, and index are kept; delete
%LOCALAPPDATA%\sems to remove them too.

.EXAMPLE
.\tools\install\install.ps1
.\tools\install\install.ps1 -Cuda
.\tools\install\install.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$SkipBuild,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$DataDirectory = Join-Path $env:LOCALAPPDATA 'sems'
$BinDirectory = Join-Path $DataDirectory 'bin'
$InstalledExecutable = Join-Path $BinDirectory 'sems.exe'

# --- User PATH -------------------------------------------------------------------------------------
# Edited in the registry as REG_EXPAND_SZ, read without expansion, so entries such as %USERPROFILE%\bin
# survive. [Environment]::SetEnvironmentVariable would rewrite them as expanded, fixed strings.

function Get-UserPathEntries {
    $raw = (Get-Item -Path 'HKCU:\Environment').GetValue('Path', '', 'DoNotExpandEnvironmentNames')
    return @($raw -split ';' | Where-Object { $_ -ne '' })
}

function Set-UserPathEntries([string[]]$Entries) {
    Set-ItemProperty -Path 'HKCU:\Environment' -Name 'Path' -Value ($Entries -join ';') -Type ExpandString
    Send-EnvironmentChangeNotification
}

# Tells Explorer and newly started terminals that the environment changed (what the Environment
# Variables dialog does), so a sign-out is not needed.
function Send-EnvironmentChangeNotification {
    if (-not ('Sems.NativeMethods' -as [type])) {
        Add-Type -Namespace Sems -Name NativeMethods -MemberDefinition @'
[DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern IntPtr SendMessageTimeout(IntPtr window, uint message, UIntPtr wParam, string lParam,
    uint flags, uint timeoutMilliseconds, out UIntPtr result);
'@
    }
    $broadcast = [IntPtr]0xffff
    $settingChange = 0x001A
    $abortIfHung = 0x0002
    $result = [UIntPtr]::Zero
    [void][Sems.NativeMethods]::SendMessageTimeout($broadcast, $settingChange, [UIntPtr]::Zero, 'Environment',
        $abortIfHung, 5000, [ref]$result)
}

function Add-ToUserPath([string]$Directory) {
    $entries = Get-UserPathEntries
    if ($entries | Where-Object { $_.TrimEnd('\') -ieq $Directory.TrimEnd('\') }) {
        return $false
    }
    Set-UserPathEntries ($entries + $Directory)
    return $true
}

function Remove-FromUserPath([string]$Directory) {
    $entries = Get-UserPathEntries
    $kept = @($entries | Where-Object { $_.TrimEnd('\') -ine $Directory.TrimEnd('\') })
    if ($kept.Count -eq $entries.Count) {
        return $false
    }
    Set-UserPathEntries $kept
    return $true
}

# --- Components ------------------------------------------------------------------------------------

function Find-Cargo {
    $cargo = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cargo) { return $cargo.Source }
    $userCargo = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
    if (Test-Path $userCargo) { return $userCargo }
    throw 'cargo was not found. Install Rust from https://rustup.rs, or pass -SkipBuild with a built target\release\sems.exe.'
}

function Install-Executable {
    $built = Join-Path $RepositoryRoot 'target\release\sems.exe'
    if (-not $SkipBuild) {
        Write-Host 'Building sems (release)...'
        & (Find-Cargo) build --release --manifest-path (Join-Path $RepositoryRoot 'Cargo.toml')
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path $built)) { throw "No built executable at $built; run without -SkipBuild." }
    New-Item -ItemType Directory -Force -Path $BinDirectory | Out-Null
    try {
        Copy-Item -Path $built -Destination $InstalledExecutable -Force
    } catch {
        throw "Could not replace $InstalledExecutable (is sems still running?): $($_.Exception.Message)"
    }
    Write-Host "Installed $InstalledExecutable"
}

function Install-CudaPack {
    & $InstalledExecutable gpu install
    if ($LASTEXITCODE -ne 0) { throw "Installing the CUDA pack failed with exit code $LASTEXITCODE" }
}

# sems runs the user's ffmpeg to decode audio and video rather than bundling one.
function Show-FfmpegStatus {
    if ($env:SEMS_FFMPEG -or (Get-Command 'ffmpeg' -ErrorAction SilentlyContinue)) {
        Write-Host 'ffmpeg found: audio and video will be indexed'
    } else {
        Write-Host 'ffmpeg not found: audio and video are skipped until it is installed (winget install Gyan.FFmpeg)'
    }
}

# --- Main ------------------------------------------------------------------------------------------

if ($Uninstall) {
    if (Test-Path $InstalledExecutable) {
        Remove-Item -Path $InstalledExecutable -Force
        Write-Host "Removed $InstalledExecutable"
    }
    if (Remove-FromUserPath $BinDirectory) { Write-Host "Removed $BinDirectory from your PATH" }
    Write-Host "Kept the model, runtimes, and index in $DataDirectory."
    return
}

Install-Executable
if ($Cuda) { Install-CudaPack }
Show-FfmpegStatus

if (Add-ToUserPath $BinDirectory) {
    Write-Host "Added $BinDirectory to your PATH. Open a new terminal to use 'sems'."
} else {
    Write-Host "$BinDirectory is already on your PATH."
}
$version = & $InstalledExecutable --version
Write-Host "Done: $version"
