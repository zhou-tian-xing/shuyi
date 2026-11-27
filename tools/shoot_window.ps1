# Capture a window to a PNG, so the UI can be reviewed from a script.
#
#   pwsh -File tools/shoot_window.ps1 -ProcessName shuyi -Out shot.png
#
# PrintWindow (not a screen grab) so it also works when the window is behind
# others or the session is not interactive.
param(
    [string]$ProcessName = "shuyi",
    [string]$Out = "shot.png",
    [int]$WaitMs = 1500
)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinShot {
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool UpdateWindow(IntPtr h);
}
"@
$proc = Get-Process -Name $ProcessName -ErrorAction Stop | Select-Object -First 1
$handle = $proc.MainWindowHandle
if ($handle -eq [IntPtr]::Zero) { throw "no window for $ProcessName" }
Start-Sleep -Milliseconds $WaitMs
[void][WinShot]::SetForegroundWindow($handle)
[void][WinShot]::UpdateWindow($handle)
Start-Sleep -Milliseconds 300
$rect = New-Object WinShot+RECT
[void][WinShot]::GetWindowRect($handle, [ref]$rect)
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$hdc = $graphics.GetHdc()
[void][WinShot]::PrintWindow($handle, $hdc, 2)
$graphics.ReleaseHdc($hdc)
$graphics.Dispose()
$bitmap.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bitmap.Dispose()
Write-Output "saved $Out ($width x $height)"
