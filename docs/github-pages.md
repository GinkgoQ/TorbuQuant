# GitHub Pages

This repository deploys the MkDocs Material site from the `main` branch with
GitHub Actions. It should not use a `gh-pages` branch and it should not use the
GitHub Pages "Deploy from a branch" mode.

The Pages source in repository settings must be:

```text
Build and deployment -> Source -> GitHub Actions
```

If Pages is set to `main` / `docs`, GitHub Pages runs Jekyll against the
Markdown source files in `docs/`. That produces the plain Primer-style page and
can fail with errors like:

```text
github-pages | Error: No such file or directory @ dir_chdir0 - /github/workspace/docs
```

The correct path is:

```text
main push -> Deploy docs workflow -> mkdocs build --strict -> deploy-pages artifact
```

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

## Deployment

The repository workflow lives at `.github/workflows/docs.yml`.

It:

- runs on pushes to `main` and manual dispatches,
- installs the docs extra,
- builds the site into `site/`,
- adds `.nojekyll` to the artifact,
- deploys with `actions/deploy-pages`.

The Python package publishing workflow is separate:
`.github/workflows/publish.yml` builds and publishes to PyPI only for `v*` tags
or manual dispatches.
