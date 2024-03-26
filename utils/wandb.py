"""
Implements some functions to retrieve data from wandb.
"""

import wandb
from experiments.training import create_tasks


def retrieve_wandb_runs(run_ids=None, group=None):
    """
    For a set of W&B run ids, retrieve the corresponding runs, and
    returns their configurations.

    Parameters
    ----------
    run_ids : list of str, optional
        The ids of the runs to retrieve.
    group: str, optional
        The group of runs to retrieve. Exactly one of `run_ids` and `group`
        must be provided.

    Returns
    -------
    runs: Mapping run_id -> run
    configs: Mapping run_id -> config
    tasks: Mapping run_id -> Mapping task_name -> task dict
    """
    if run_ids is None and group is None:
        raise ValueError("Either `run_ids` or `group` must be provided.")
    if run_ids is not None and group is not None:
        raise ValueError("Only one of `run_ids` and `group` should be provided.")
    api = wandb.Api()
    # If a group is provided, retrieve the runs in the group
    if group is not None:
        runs = api.runs("arches/tc_prediction", filters={"group": group})
        run_ids = [run.id for run in runs]
    # Otherwise, retrieve the runs with the provided ids
    else:
        runs = {}
        for run_id in run_ids:
            runs[run_id] = api.run(f"arches/tc_prediction/{run_id}")
    # Retrieve the configurations
    configs, tasks = {}, {}
    for run_id in run_ids:
        configs[run_id] = runs[run_id].config

        # Backward compatibility: replace the "future_steps" key by
        # "target_steps":
        exp_cfg = configs[run_id]["experiment"]
        if "future_steps" in exp_cfg:
            exp_cfg["target_steps"] = [t for t in range(1, exp_cfg["future_steps"] + 1)]
            if "perform_estimation" in exp_cfg:
                exp_cfg["target_steps"] += [-t for t in range(0, exp_cfg["past_steps"])]

        # Retrieve the tasks performed by the run
        tasks[run_id] = create_tasks(configs[run_id])
    return runs, configs, tasks
