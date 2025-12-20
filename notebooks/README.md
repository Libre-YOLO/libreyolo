# Libre YOLO Notebooks

This directory contains Jupyter notebooks designed for researchers and developers to explore the architecture, perform inference, and experiment with Libre YOLO models.

These notebooks provide a hands-on way to understand how the models are constructed and how to interact with the Libre YOLO API.

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| [**libreyolo8.ipynb**](libreyolo8.ipynb) | Architecture exploration, layer inspection, and inference examples for **YOLOv8**. Ideal for understanding the v8 backbone and head structure. |
| [**libreyolo11.ipynb**](libreyolo11.ipynb) | Deep dive into the **YOLOv11** architecture. Demonstrates the new features and improvements in v11 compared to previous versions. |

## Usage

To run these notebooks, ensure you have the development dependencies installed (including `jupyter` or `jupyterlab`).

```bash
# From the project root
make setup
```

Then launch Jupyter:

```bash
uv run jupyter lab notebooks/
```
