#!/bin/bash

#pytest-html

project="llmada"
uv run pytest --html=$test_html_path/$project.html
