"""
Implements some functions to retrieve data from wandb.
"""

import wandb
from experiments.training import create_tasks


def retrieve_wandb_runs(run_ids):
    """
    For a set of W&B run ids, retrieve the corresponding runs, and
    returns their configurations.

    Parameters
    ----------
    run_ids : list of str
        The ids of the runs to retrieve.

    Returns
    -------
    runs: Mapping run_id -> run
    configs: Mapping run_id -> config
    tasks: Mapping run_id -> Mapping task_name -> task dict
    """
    api = wandb.Api()
    configs, runs, tasks = {}, {}, {}
    for run_id in run_ids:
        try:
            run = api.run(f"arches/tc_prediction/{run_id}")
        except wandb.errors.CommError:
            print(f"WARNING: Run {run_id} could not be retrieved.")
        runs[run_id] = run
        configs[run_id] = run.config

        # Backward compatibility: replace the "future_steps" key by
        # "target_steps":
        exp_cfg = configs[run_id]['experiment']
        if 'future_steps' in exp_cfg:
            exp_cfg['target_steps'] = [t for t in range(1, exp_cfg['future_steps'] + 1)]
            if 'perform_estimation' in exp_cfg:
                exp_cfg['target_steps'] += [-t for t in range(0, exp_cfg['past_steps'])]

        # Retrieve the tasks performed by the run
        tasks[run_id] = create_tasks(configs[run_id])
    return runs, configs, tasks
