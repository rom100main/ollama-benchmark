# Benchmark Ollama models inference speed

## Prerequisites

[Ollama](https://ollama.com)  
[Python](https://www.python.org)

## Installation

### Create a virtual environment
```bash
python -m venv venv
```

### Activate the virtual environment

on Linux/MacOS
```bash
source venv\Scripts\activate
```

on Windows
```cmd
venv\Scripts\activate
```

### Install the dependencies
```bash
pip install -r requirements.txt
```

## Run

```bash
python speed.py -m llama3.1:latest
```

## Help

```bash
python speed.py -h
```
