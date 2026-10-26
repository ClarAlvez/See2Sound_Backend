@echo off
setlocal
if not exist "API\main.py" goto wrong_root
if not exist "pipeline\orchestration\verified_pipeline.py" goto wrong_root
if not exist "%~dp0API\main.py" goto wrong_root
for %%I in (".") do set "CURRENT_ROOT=%%~fI"
for %%I in ("%~dp0.") do set "SCRIPT_ROOT=%%~fI"
if /I not "%CURRENT_ROOT%"=="%SCRIPT_ROOT%" goto wrong_root

set "SEE2SOUND_APPLY_ACTION_CORRECTIONS=true"
set "SEE2SOUND_PERSON_INTERMEDIATE=true"
set "SEE2SOUND_OBJECT_INTERMEDIATE=true"
set "SEE2SOUND_OBJECT_SUPPRESS_NARRATIVE=true"
set "SEE2SOUND_OBJECT_APPLY_ADDITIONS=false"
set "SEE2SOUND_OBJECT_APPLY_REMOVALS=false"
set "SEE2SOUND_TTS_PROVIDER=edge"
set "SEE2SOUND_TTS_VOICE=pt-BR-FranciscaNeural"
set "SEE2SOUND_TTS_RATE=-5%%"
set "SEE2SOUND_TTS_PITCH=+0Hz"

set "DEMO_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "DEMO_PYTHON=%CD%\.venv\Scripts\python.exe"
"%DEMO_PYTHON%" --version
if errorlevel 1 goto python_failed
echo Perfil de demonstracao: Actions=true Person=true Objects=true
echo Objects: suppress=true additions=false removals=false
echo TTS: edge / pt-BR-FranciscaNeural / rate=-5%% / pitch=+0Hz
echo Executando diagnostico local sem carregar modelos...
"%DEMO_PYTHON%" -B tools\check_backend.py --skip-api --require-port-free
if errorlevel 1 goto diagnostic_failed
echo API local: http://127.0.0.1:8000 -- um worker, sem reload.
echo Ctrl+C encerra a API. As variaveis deste script sao locais e serao restauradas.
"%DEMO_PYTHON%" -m uvicorn API.main:app --host 127.0.0.1 --port 8000 --workers 1
if errorlevel 1 goto api_failed
exit /b 0

:wrong_root
echo ERRO: execute start_demo.cmd na raiz deste repositorio See2Sound_Backend.
exit /b 1
:python_failed
echo ERRO: Python indisponivel. Verifique .venv\Scripts\python.exe e o Python base.
exit /b 1
:diagnostic_failed
echo ERRO: diagnostico reprovado. Corrija os itens ERRO acima; nenhum pacote foi instalado.
exit /b 1
:api_failed
echo ERRO: a API encerrou com falha. Consulte o traceback acima, a porta 8000 e o uso de um unico worker.
exit /b 1
