Option Explicit

Dim shell, fso, rootDir, batPath

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = fso.BuildPath(rootDir, "run_omnispeech.bat")

If Not fso.FileExists(batPath) Then
  MsgBox "OmniSpeech baslatma scripti bulunamadi:" & vbCrLf & batPath, vbExclamation, "OmniSpeech"
  WScript.Quit 1
End If

shell.CurrentDirectory = rootDir
shell.Run "cmd /c """ & batPath & """", 1, False
