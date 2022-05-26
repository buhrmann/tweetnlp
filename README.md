# Install

## PyPi

Simply run

```
pip install tweetnlp
```

## Development

To install a local copy for development, including all dependencies for test, documentation and code quality, use the following commands:

``` bash
clone git+https://github.com/cardiffnlp/tweetnlp
cd tweetnlp
pip install -v -e ".[dev]"
pre-commit install
```

The [pre-commit](https://pre-commit.com/) command will make sure that whenever you try to commit changes to this repo code quality and formatting tools will be executed. This ensures e.g. a common coding style, such that any changes to be commited are functional changes only, not changes due to different personal coding style preferences. This in turn makes it either to collaborate via pull requests etc.

To test installation you may execute the [pytest](https://docs.pytest.org/en/7.1.x/) suite to make sure everything's setup correctly, e.g.:

``` bash
pytest -v .
```

# Command line interface

To see the available commandos call `tweetnlp --help`:

``` bash
> tweetnlp --help

Usage: tweetnlp [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.

Commands:
  score-model  Evaluate a model against a task's test labels.
  score-pred   Evaluate a predictions file against a task's test labels.
```

If you have a file with predictions for a specific task, `tweetnlp score-pred` prints the corresponding score. For example, the following should return a perfect score of 1.0:

```
> tweetnlp score-pred emoji tweetnlp/resources/datasets/emoji/test_labels.txt
1.0
```

To evaluate a Cardiff model pretrained on a specific task you can use

```
tweetnlp score-model emoji
```

Support for arbitrary huggingface classifieres coming up...

# Notebook

If you installed this packages into a brand new environment, and depending on the kind of environment (venv, conda etc.), after installation you may have to register this environment with jupyter for it to show as a kernel in notebooks:

``` bash
ipython kernel install --name [myenv] --user
```

Following this, start jupyter with `jupyter notebook` and it should let you select the kernel containing your tweetnlp installation.
