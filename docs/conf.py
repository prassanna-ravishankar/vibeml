"""Sphinx configuration."""

project = "vibeml"
author = "Prass, The Nomadic Coder"
copyright = "2025, Prass, The Nomadic Coder"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "shibuya"
