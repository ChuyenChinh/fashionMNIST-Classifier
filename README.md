# FashionMNIST

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An mini project of neural network based classifier on Fashion MNIST dataset

## Project Organization

```
.
|-- Makefile              <- Convenience commands for project tasks.
|-- README.md             <- Project overview and usage notes.
|-- pyproject.toml        <- Project metadata and Python dependencies.
|-- setup.cfg             <- Tooling configuration.
|-- uv.lock               <- Locked dependency versions for uv.
|-- data
|   |-- external          <- Data from third-party sources.
|   |-- interim           <- Intermediate transformed data.
|   |-- processed         <- Final datasets prepared for modeling.
|   |-- raw               <- Original FashionMNIST downloads.
|-- docs                  <- Project documentation.
|-- models                <- Saved model weights, such as `fashion_mlp.pth`.
|-- notebooks
|   `-- demo.ipynb        <- Notebook for interactive exploration or demos.
|-- references            <- Supporting notes, manuals, or reference material.
|-- reports
|   `-- figures           <- Generated figures for reports.
`-- src
    |-- __init__.py       <- Marks `src` as a Python package.
    |-- config.py         <- Shared paths and training hyperparameters.
    |-- dataset.py        <- FashionMNIST dataset and DataLoader setup.
    |-- features.py       <- Torchvision transforms for preprocessing.
    `-- modeling
        |-- __init__.py
        |-- network.py    <- MLP model definition.
        |-- predict.py    <- Reserved for model inference code.
        `-- train.py      <- Training loop and model checkpoint saving.
```

