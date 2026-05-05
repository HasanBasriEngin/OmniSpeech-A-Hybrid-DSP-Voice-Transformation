Option Explicit

Dim shell, fso, rootDir, exePath

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
exePath = fso.BuildPath(rootDir, "src-tauri\target\debug\omnispeech_desktop.exe")

If Not fso.FileExists(exePath) Then
  MsgBox "OmniSpeech exe bulunamadi:" & vbCrLf & exePath, vbExclamation, "OmniSpeech"
  WScript.Quit 1
End If

shell.CurrentDirectory = rootDir
shell.Run """" & exePath & """", 1, False
