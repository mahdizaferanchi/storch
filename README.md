# storch

Pytorch starter project containing training code and utils like configuration and logging.

### Usage notes

- The configurable decorator should be used to make functions / classes configurable from a global configuration object.
  - Should be possible to change important configuration options from the command line using `argparse`
- Runs should be optionally trackable with W&B and in that case the trainer state should be saved at the end of training (and loaded at the beginning).
- We recommend using `torchmetrics` for metrics.
- Please see the file `language_model.py` in the examples folder to see an example of how to use this library.
