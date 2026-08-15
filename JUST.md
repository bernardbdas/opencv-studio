# Justfile Task Runner Guide 🛠️

This project uses `just` as a command runner. It simplifies execution of python setups, docker container orchestration, cleaning, and testing.

## Installation

If you don't have `just` installed on your system, you can install it using your system's package manager:

- **macOS (Homebrew)**: `brew install just`
- **Linux (Debian/Ubuntu)**: `sudo apt install just`
- **Linux (Arch)**: `sudo pacman -S just`
- **Windows (Chocolatey)**: `choco install just`
- **Python/Pip (Generic)**: `pip install rust-just`

---

## Command Reference

Run the commands below from the project root.

| Command | Action | Explanation |
|---|---|---|
| `just build` | `uv sync` | Prepares the Python virtual environment (`.venv`) and synchronizes all dependencies listed in `pyproject.toml`. |
| `just run` | `uv run python main.py` | Launches the Streamlit computer vision app locally (default port `8501`). |
| `just lint` | Syntax check | Compiles python files across the repository to ensure there are no syntax errors. |
| `just docker-build` | `docker compose build` | Builds the Docker image containing the MediaPipe solutions portal. |
| `just docker-run` | `docker compose up` | Launches the containerized portal at `http://localhost:8080` with pre-installed system dependencies (OpenGL drivers, etc.). |
| `just docker-stop` | `docker compose down` | Stops and tears down the active Docker containers. |
| `just clean` | Cache cleanup | Cleans temporary compilation output, `__pycache__` directories, and `.pyc` files. |

---

## Basic Usage Examples

### 1. Initial Project Scaffolding
Get the dependencies installed and run the application:
```bash
just build
just run
```

### 2. Sandbox Testing with Docker
To test compilation and performance in an isolated Linux container:
```bash
just docker-build
just docker-run
```
Once done, clean up with:
```bash
just docker-stop
```
