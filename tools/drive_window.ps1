# Drive the 数弈 window from a script: clicks, keys, and a screenshot.
#
#   pwsh -File tools/drive_window.ps1 -ProcessName shuyi -Actions "click:644,486;wait:2500;shot:build/shot.png"
#
# Actions:
#   click:X,Y   left click at window-client coordinates X,Y (the app's own pixels)
#   key:NAME    send WM_KEYDOWN for a virtual key (SPACE, UP, DOWN, DELETE, 1..9)
#   wait:MS     sleep
#   shot:PATH   capture the window to PATH
param(
    [string]$ProcessName = "shuyi",
    [Parameter(Mandatory = $true)][string]$Actions
)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Drive {
    [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool ScreenToClient(IntPtr h, ref POINT p);
    [DllImport("user32.dll")] public static extern IntPtr SendMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
}
"@
$proc = Get-Process -Name $ProcessName -ErrorAction Stop | Select-Object -First 1
$h = $proc.MainWindowHandle
if ($h -eq [IntPtr]::Zero) { throw "no window for $ProcessName" }
[void][Drive]::SetForegroundWindow($h)

function Shot([string]$path) {
    $rect = New-Object Drive+RECT
    [void][Drive]::GetWindowRect($h, [ref]$rect)
    $w = $rect.Right - $rect.Left
    $ht = $rect.Bottom - $rect.Top
    $bmp = New-Object System.Drawing.Bitmap $w, $ht
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $dc = $g.GetHdc()
    [void][Drive]::PrintWindow($h, $dc, 2)
    $g.ReleaseHdc($dc)
    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Output "shot $path"
}

$keys = @{ SPACE = 0x20; UP = 0x26; DOWN = 0x28; DELETE = 0x2E; LEFT = 0x25; RIGHT = 0x27 }
for ($i = 0; $i -lt 9; $i++) { $keys["$($i + 1)"] = 0x31 + $i }

foreach ($action in $Actions.Split(';')) {
    if ($action -eq '') { continue }
    $parts = $action.Split(':')
    switch ($parts[0]) {
        'click' {
            $xy = $parts[1].Split(',')
            # Client coordinates: the window proc receives exactly what we send, so
            # there is no screen-origin arithmetic to get wrong.
            $p = New-Object Drive+POINT
            $p.X = [int]$xy[0]
            $p.Y = [int]$xy[1]
            $lp = [IntPtr](($p.Y -shl 16) -bor ($p.X -band 0xFFFF))
            [void][Drive]::SendMessageW($h, 0x0201, [IntPtr]1, $lp)   # WM_LBUTTONDOWN
            [void][Drive]::SendMessageW($h, 0x0202, [IntPtr]0, $lp)   # WM_LBUTTONUP
            Write-Output "click $($xy[0]),$($xy[1])"
        }
        'key' {
            $vk = $keys[$parts[1].ToUpper()]
            if ($null -eq $vk) { throw "unknown key $($parts[1])" }
            [void][Drive]::SendMessageW($h, 0x0100, [IntPtr]$vk, [IntPtr]0)
            Write-Output "key $($parts[1])"
        }
        # A menu command, by the id the menu puts in WM_COMMAND.
        'cmd' {
            [void][Drive]::SendMessageW($h, 0x0111, [IntPtr][int]$parts[1], [IntPtr]0)
            Write-Output "cmd $($parts[1])"
        }
        # Bring the window back without taking the focus: PrintWindow needs it to
        # exist on screen, and a capture must not interrupt whoever is working.
        'show' {
            [void][Drive]::ShowWindow($h, 4)   # SW_SHOWNOACTIVATE
            Write-Output "show"
        }
        'wait' { Start-Sleep -Milliseconds ([int]$parts[1]) }
        'shot' { Shot $parts[1] }
        default { throw "unknown action $action" }
    }
}
