"""
Implements some functions to retrieve data from wandb.
"""

import wandb
import yaml
from utils.utils import update_dict
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
    # Retrieve the base config
    with open("training_cfg.yml", "r") as file:
        base_cfg = yaml.safe_load(file)

    api = wandb.Api()
    configs, runs, tasks = {}, {}, {}
    for run_id in run_ids:
        try:
            run = api.run(f"arches/tc_prediction/{run_id}")
        except wandb.errors.CommError:
            print(f"WARNING: Run {run_id} could not be retrieved.")

        # The returned cfg is the base config, overwritten by the run config
        configs[run_id] = update_dict(base_cfg, run.config)
        runs[run_id] = run

        # Retrieve the tasks performed by the run
        tasks[run_id] = create_tasks(configs[run_id])
    return runs, configs, tasks
