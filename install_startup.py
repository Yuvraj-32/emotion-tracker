"""
==========================================================
  EMOTION TRACKER — STARTUP INSTALLER
==========================================================
  Adds or removes the Emotion Tracker from Windows Startup.

  Usage:
    python install_startup.py              → Install
    python install_startup.py --uninstall  → Uninstall
==========================================================
"""

import os
import sys
import argparse


def get_startup_folder():
    """Get the Windows Startup folder path."""
    return os.path.join(
        os.environ.get('APPDATA', ''),
        'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup'
    )


def get_launcher_path():
    """Get the absolute path to launcher.pyw."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'launcher.pyw')


def create_shortcut(target, shortcut_path):
    """Create a Windows .lnk shortcut using PowerShell."""
    # Use PowerShell COM object to create proper .lnk shortcut
    ps_script = f'''
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("{shortcut_path}")
$sc.TargetPath = "{sys.executable.replace('python.exe', 'pythonw.exe')}"
$sc.Arguments = '"{target}"'
$sc.WorkingDirectory = "{os.path.dirname(target)}"
$sc.Description = "Emotion & Attention Tracker"
$sc.Save()
'''
    import subprocess
    result = subprocess.run(
        ['powershell', '-Command', ps_script],
        capture_output=True, text=True
    )
    return result.returncode == 0


def install():
    """Install the launcher to Windows Startup."""
    startup = get_startup_folder()
    launcher = get_launcher_path()
    shortcut = os.path.join(startup, 'EmotionTracker.lnk')

    if not os.path.exists(launcher):
        print(f"ERROR: launcher.pyw not found at: {launcher}")
        return False

    print("=" * 50)
    print("  EMOTION TRACKER — STARTUP INSTALLER")
    print("=" * 50)
    print(f"  Launcher:  {launcher}")
    print(f"  Startup:   {startup}")
    print(f"  Shortcut:  {shortcut}")
    print()

    if create_shortcut(launcher, shortcut):
        print("  ✅ SUCCESS! Emotion Tracker will start on login.")
        print()
        print("  The tracker will run in the system tray (🧠)")
        print("  Right-click the tray icon for controls.")
        print("=" * 50)
        return True
    else:
        print("  ❌ FAILED to create shortcut.")
        print("  Try running this script as Administrator.")
        print("=" * 50)
        return False


def uninstall():
    """Remove the launcher from Windows Startup."""
    startup = get_startup_folder()
    shortcut = os.path.join(startup, 'EmotionTracker.lnk')

    print("=" * 50)
    print("  EMOTION TRACKER — STARTUP UNINSTALLER")
    print("=" * 50)

    if os.path.exists(shortcut):
        os.remove(shortcut)
        print("  ✅ Startup shortcut removed.")
        print("  Emotion Tracker will no longer start on login.")
    else:
        print("  ℹ️  No startup shortcut found (already removed).")

    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Install/uninstall Emotion Tracker startup')
    parser.add_argument('--uninstall', action='store_true',
                        help='Remove from Windows Startup')
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
    else:
        install()
