# wandb-log-bug

This repository provides a simple example that produces a [logging error](https://github.com/wandb/client/issues/1314) when using Wandb logger with Hydra multiruns.

To reproduce the error:
- **Step 0**: Clone this repository, create a new environment, and install the requirements with
    ```
    pip install -r requirements.txt
    ```

    You also need to create a Wandb project named `wandb-log-bug`.

- **Step 1**: Run a Hydra multirun with
    ```
    python main.py -m learning_rate=1e-3,1e-4
    ```

- **Step 2**: Wait ~8 seconds (length of one run) and observe the bug. In my case the following error appears:
    ```python
    --- Logging error ---
    Traceback (most recent call last):
    File "/home/tadej/conda/lib/python3.8/logging/__init__.py", line 1085, in emit
        self.flush()
    File "/home/tadej/conda/lib/python3.8/logging/__init__.py", line 1065, in flush
        self.stream.flush()
    ValueError: I/O operation on closed file.
    Call stack:
    File "main.py", line 20, in <module>
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
    File "main.py", line 17, in run_train
        trainer.fit(model, data_module)
    File "/home/tadej/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py", line 454, in fit
        results = self.accelerator_backend.train()
    File "/home/tadej/conda/lib/python3.8/site-packages/pytorch_lightning/accelerators/cpu_backend.py", line 44, in train
        self.trainer.train_loop.setup_training(model)
    File "/home/tadej/conda/lib/python3.8/site-packages/pytorch_lightning/trainer/training_loop.py", line 151, in setup_training
        ref_model.summarize(mode=self.trainer.weights_summary)
    File "/home/tadej/conda/lib/python3.8/site-packages/pytorch_lightning/core/lightning.py", line 1215, in summarize
        log.info("\n" + str(model_summary))
    Message: '\n  | Name    | Type      | Params\n--------------------------------------\n0 | conv    | Conv2d    | 80    \n1 | dropout | Dropout2d | 0     \n2 | fc      | Linear    | 2 K   \n3 | pool    | MaxPool2d | 0     '
    Arguments: ()
    Traceback (most recent call last):                                                                                                          
    File "/home/tadej/conda/lib/python3.8/site-packages/hydra/core/utils.py", line 247, in _flush_loggers
        h_weak_ref().flush()
    File "/home/tadej/conda/lib/python3.8/logging/__init__.py", line 1065, in flush
        self.stream.flush()
    ValueError: I/O operation on closed file.

    Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
    ```
