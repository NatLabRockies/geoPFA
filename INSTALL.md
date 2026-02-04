# Installing NREL's Geothermal Play Fairway Analysis

There are several ways to install geoPFA. We strongly recommend using a
virtual environment to avoid conflicts with other packages and projects
on your system.

## Quick Start

Choose your preferred installation method:

1. [**Pixi** (recommended for development)](#installing-with-pixi-recommended) - Reproducible environments with exact dependency versions
2. [**pip**](#installing-with-pip) - Standard Python package installation
3. [**Conda/Mamba**](#installing-with-conda) - Conda ecosystem integration
4. [**Wheels**](#installing-from-released-wheels) - Pre-built distribution files

---

## Installing with Pixi (recommended)

[Pixi](https://pixi.sh) provides reproducible Python environments and is our recommended tool for development. It uses the `pyproject.toml` and `pixi.lock` files to ensure you have exactly the same dependencies as the development team.

### Installation Steps

1. **Install Pixi** (skip if already installed)

   **Linux and macOS:**
   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

   **Windows (PowerShell):**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
   ```

   For more installation options, see [pixi.sh](https://pixi.sh).

2. **Clone the repository** (if you haven't already)

   ```bash
   git clone https://github.com/NatLabRockies/geoPFA
   cd geoPFA
   ```

3. **Activate the Pixi environment**

   ```bash
   pixi shell
   ```

   You should see a `(geoPFA)` prefix in your terminal prompt, indicating the
   environment is active.

4. **Verify the installation**

   Inside Python:
   ```python
   import geopfa
   print(geopfa.__version__)
   ```

---

## Installing with pip

Installing with PIP is the most basic way of installing a Python package, and
it is typically used to build containers or with some environment manager such
as `virtualenv`.
`geoPFA` requires **Python 3.11 or newer**.
You might need to install PIP if it is not already provided.

Inside your preferred environment manager

1. **Upgrade pip:**

   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install geoPFA:**

   **From PyPI** (public projects available in PyPI):
   ```bash
   python -m pip install geoPFA
   ```

   **From GitHub (latest):**
   ```bash
   python -m pip install "git+https://github.com/NatLabRockies/geoPFA"
   ```

   **From a local clone:**
   ```bash
   python -m pip install .
   ```

3. **Verify the installation:**

   Inside Python:
   ```python
   import geopfa
   print(geopfa.__version__)
   ```

---

## Installing from released Wheels

Download from https://github.com/GeothermalExplorationTools/geopfa/releases

## Installing with Conda/Mamba
