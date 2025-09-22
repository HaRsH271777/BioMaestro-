@echo off
REM BioMaestro Batch Inference Script

REM --- CONFIGURATION ---
REM Set the path to the checkpoint you want to use
SET CHECKPOINT_PATH=models\biomaestro_checkpoints\biomaestro-epoch=27-train_loss=0.06.ckpt

REM Set the path to your audio files
SET AUDIO_DIR=data\rfcx-species-audio-detection\train

REM Set the path for the output plots
SET OUTPUT_DIR=results

ECHO.
ECHO Starting batch inference...
ECHO Using checkpoint: %CHECKPOINT_PATH%
ECHO.

REM Loop through all .flac files in the audio directory
FOR %%f IN ("%AUDIO_DIR%\*.flac") DO (
    ECHO Processing file: %%~nxf

    REM Define the output filename
    SET OUT_FILENAME=%%~nf_prediction.png

    REM Run the inference script
    python src/run_inference.py --checkpoint %CHECKPOINT_PATH% --input "%%f" --output "%OUTPUT_DIR%\!OUT_FILENAME!"
)

ECHO.
ECHO 🎉 Batch inference complete! All plots saved in the '%OUTPUT_DIR%' folder.