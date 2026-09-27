# Contributing

## Development

Use Python 3.10+. Linting, formatting, and package builds do not require
PyTorch or k2. Runtime installation is described in [README.md](README.md).

```bash
python -m pip install --upgrade ruff
python -m ruff check .
python -m ruff format --check .
```

CI checks linting, formatting, and package builds.

Keep the public API focused on `pruned_ctc_loss`.
Contributions use the [MIT license](LICENSE).

## Building a package

The build does not require PyTorch or k2:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check --strict dist/*
```

This produces a wheel and source distribution in `dist/`. CI also builds,
checks, and uploads these files as workflow artifacts.

## Publishing a release

Publishing is manual. The `Publish` workflow runs only through
`workflow_dispatch`; pushes and GitHub releases do not publish packages.

Before the first upload, the repository owner must configure
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) independently on
PyPI and TestPyPI and create the corresponding GitHub environments:

| Setting | TestPyPI | PyPI |
| --- | --- | --- |
| Distribution | `pruned-ctc` | `pruned-ctc` |
| Repository owner | `yfyeung` | `yfyeung` |
| Repository | `PrunedCTC` | `PrunedCTC` |
| Workflow filename | `publish.yml` | `publish.yml` |
| GitHub environment | `testpypi` | `pypi` |

A new project can use a pending Trusted Publisher. Configure environment
protection rules as appropriate for the repository. Authentication uses a
short-lived GitHub OIDC token; no PyPI password or API token is stored in the
workflow.

For each release, update `__version__` in `pruned_ctc.py`, run the checks above,
and push the reviewed revision. In GitHub Actions, select **Publish → Run
workflow** and choose that revision. Start with the default `testpypi` target;
after checking its uploaded distributions, run the workflow for `pypi` using
the same revision. Each run builds and checks its artifacts before the
publishing job enters the selected environment.
