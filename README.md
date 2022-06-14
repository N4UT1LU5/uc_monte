# Ecological Succession Analysis CLI-Tool

Simple python CLI-tool to detect ecological succession in a set of CIR tiff-orthoimages.

| Flag | Description                               |
| ---- | ----------------------------------------- |
| -i   | Path to folder with input files           |
| -o   | Path to output folder                     |
| -gsd | Ground sample distance in meter per pixel |
| -th  | NDVI vegetation threshold                 |

## Setup

Create virtual environment named succ

```shell
python3 -m venv succ
```

activate virtual enviroment

install all packages listed in pip.txt file

```shell
pip install -r pip.txt
```

save all installed packages to pip.txt file

```shell
pip freeze > pip.txt
```
