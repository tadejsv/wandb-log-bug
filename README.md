# wandb-log-bug

This repository provides a simple example that produces a [logging error](https://github.com/wandb/client/issues/1314) when using Wandb logger with Hydra multiruns.

The error is related to `reinit=True` option of wandb, if `reinit` is set to `False` the error dissapears.

To reproduce the error:
- **Step 0**: Clone this repository, create a new environment, and install the requirements with
    ```
    pip install -r requirements.txt
    ```

- **Step 1**: Run a Hydra multirun with
    ```
    python main.py -m learning_rate=1e-3,1e-4
    ```

- **Step 2**: Wait a few seconds and observe the bug. In my case the following error (multiple times) appears:
    ```python
    --- Logging error ---
    Traceback (most recent call last):
    File "/home/tadej/conda/lib/python3.8/logging/__init__.py", line 1085, in emit
        self.flush()
    File "/home/tadej/conda/lib/python3.8/logging/__init__.py", line 1065, in flush
        self.stream.flush()
    ValueError: I/O operation on closed file.
    Call stack:
    File "main.py", line 43, in <module>
        run_train()
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/main.py", line 32, in decorated_main
        _run_hydra(
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/_internal/utils.py", line 354, in _run_hydra
        run_and_report(
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/_internal/utils.py", line 198, in run_and_report
        return func()
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/_internal/utils.py", line 355, in <lambda>
        lambda: hydra.multirun(
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/_internal/hydra.py", line 136, in multirun
        return sweeper.sweep(arguments=task_overrides)
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/_internal/core_plugins/basic_sweeper.py", line 154, in sweep
        results = self.launcher.launch(batch, initial_job_idx=initial_job_idx)
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/_internal/core_plugins/basic_launcher.py", line 76, in launch
        ret = run_job(
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/core/utils.py", line 125, in run_job
        ret.return_value = task_function(task_cfg)
    File "main.py", line 37, in run_train
        log.info('loss %s', loss)
    Message: 'loss %s'
    Arguments: (tensor(0.4119, grad_fn=<NllLossBackward>),)
    ```
