# GitHub Pages

This repository uses MkDocs Material and mkdocstrings.

## Local Build

```bash
python -m pip install -e ".[docs]"
python -m mkdocs build
```

## Local Preview

```bash
python -m mkdocs serve
```

Open:

```text
http://127.0.0.1:8000
```

## Deploy With MkDocs

For a repository with push access:

```bash
python -m mkdocs gh-deploy
```

This writes the built site to the `gh-pages` branch.

## GitHub Actions Example

```yaml
name: docs

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: python -m pip install -e ".[docs]"
      - run: python -m mkdocs gh-deploy --force
```

Set repository Pages source to the `gh-pages` branch in GitHub settings.

