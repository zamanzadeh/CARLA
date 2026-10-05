@echo off
REM ============================================================
REM  run_msl.bat  —  Jalankan CARLA pada dataset MSL (Windows)
REM
REM  Cara pakai:
REM    run_msl.bat              -> proses semua channel MSL
REM    run_msl.bat C-1         -> proses satu channel tertentu
REM    run_msl.bat C-1 M-1 P-3 -> proses beberapa channel
REM ============================================================

setlocal ENABLEDELAYEDEXPANSION

REM --- Aktifkan virtual environment CARLA ---
call %~dp0.carla\Scripts\activate.bat

REM --- Pindah ke direktori proyek ---
cd /d %~dp0

REM --- Tentukan daftar channel yang akan diproses ---
if "%~1"=="" (
    REM Semua channel MSL
    set CHANNELS=A-1 A-2 A-3 A-4 A-5 A-6 A-7 A-8 A-9 B-1 C-1 C-2 D-1 D-11 D-12 D-13 D-14 D-15 D-16 D-2 D-3 D-4 D-5 D-6 D-7 D-8 D-9 E-1 E-10 E-11 E-12 E-13 E-2 E-3 E-4 E-5 E-6 E-7 E-8 E-9 F-1 F-2 F-3 F-4 F-5 F-7 F-8 G-1 G-2 G-3 G-4 G-6 G-7 M-1 M-2 M-3 M-4 M-5 M-6 M-7 P-1 P-10 P-11 P-14 P-15 P-2 P-3 P-4 P-7 R-1 S-1 S-2 T-1 T-10 T-12 T-13 T-2 T-3 T-4 T-5 T-8 T-9
) else (
    set CHANNELS=%*
)

REM --- Proses setiap channel ---
for %%C in (%CHANNELS%) do (
    echo.
    echo ========================================
    echo  Channel: %%C
    echo ========================================

    echo [1/2] Pretext stage ...
    python carla_pretext.py ^
        --config_env configs\env.yml ^
        --config_exp configs\pretext\carla_pretext_msl.yml ^
        --fname %%C
    if errorlevel 1 (
        echo [ERROR] Pretext gagal untuk channel %%C. Lanjut ke channel berikutnya.
        goto :next_%%C
    )

    echo [2/2] Classification stage ...
    python carla_classification.py ^
        --config_env configs\env.yml ^
        --config_exp configs\classification\carla_classification_msl.yml ^
        --fname %%C
    if errorlevel 1 (
        echo [ERROR] Classification gagal untuk channel %%C.
    )

    :next_%%C
)

echo.
echo ============================================================
echo  Selesai. Hasil tersimpan di: results\MSL\
echo ============================================================
endlocal
