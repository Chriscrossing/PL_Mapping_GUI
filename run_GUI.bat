@echo off

%WINDIR%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\miniconda3' ; conda activate PL_GUI ; python Lab_CTRL_GUI.py"

pause