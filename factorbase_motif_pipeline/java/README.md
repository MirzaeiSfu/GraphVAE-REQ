# Java Bundle for `bayes.jar`

This folder contains a self-contained Java setup used to launch `bayes.jar`
without requiring a separate Java installation on the target machine.

## Purpose

`bayes.jar` is used to visualize the Bayesian network output produced by the
FactorBase pipeline. In this project, it is intended for opening and inspecting
the Bayesian network files generated from FactorBase results, such as the
`Bif_*.xml` outputs.

The `bayes.jar` file used here is from the UBC AIspace project:
<https://aispace.org>

## Contents

- `bayes.jar`: the Bayesian network viewer application.
- `jdk-17/`: a bundled standalone Java runtime/JDK so the viewer can run even if
  Java is not installed system-wide.
- `runbayes.bat`: a Windows helper script that launches `bayes.jar`.

## How `runbayes.bat` Works

The batch file currently contains:

```bat
@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "JAVA_PATH=%SCRIPT_DIR%jdk-17\bin\java.exe"
"%JAVA_PATH%" -jar "%SCRIPT_DIR%bayes.jar"
pause
```

This launcher now uses a path relative to the batch file location, so it can run
the bundled JDK directly from this folder without requiring Java to be installed
at a fixed system path.

## Typical Use

1. Generate FactorBase output, including the Bayesian network XML files.
2. Launch `bayes.jar` with the bundled Java runtime.
3. Open the generated `Bif_*.xml` file in the viewer to inspect the learned
   Bayesian network.

## Notes

- The bundled `jdk-17` in this folder is a Windows-oriented distribution, as
  shown by the `.exe` launchers and `.dll` files.
- `runbayes.bat` is provided as a convenience launcher for Windows users.
- If you already have a working Java installation, you can also run the viewer
  manually with:

```bash
java -jar bayes.jar
```
