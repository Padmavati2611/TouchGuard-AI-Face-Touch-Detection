import subprocess
import os

powershell = os.path.join(
    os.environ.get("WINDIR", r"C:\Windows"),
    "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
)

script = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Volume = 100
$s.Speak('Warning! Do not touch your face!')
$s.Dispose()
"""

subprocess.run([powershell, "-NoProfile", "-Command", script])
