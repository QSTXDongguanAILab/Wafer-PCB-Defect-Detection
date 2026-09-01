@echo off
rem Double-click entry point. Real logic lives in run.bat (ASCII name so it
rem also works when invoked from shells whose code page mangles CJK names).
call "%~dp0run.bat" %*
