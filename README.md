# SRWS Trial Run

Reconstruct Srws data

Author: Ashim Giyanani

## Description

### Overview

This is just a basic README template. Add information as you see fit.

The example program contained therein takes multiple integers from the command line, and either
calculates their maximum, or their sum if the command line switch `-s` is given.

### Dependencies

This project uses a Conda environment. It pulls the packages from conda-forge by default.

The Python version used is 3.9.17.

To create the environment locally, use `conda` or `mamba`:

```bash
conda env create -f environment.yml
# or
mamba env create -f environment.yml
```

It is recommended to use mamba over conda, because mamba is much faster. You can install mamba in your base env:

```bash
conda install -c conda-forge mamba
```

### Tests

The example project herein uses `pytest` and it is configured, as per [conftest.py](conftest.py), that your main program
resides in [srws_trial_run](srws_trial_run).

To run the tests, switch to your environment, install `pytest` if not already done, and run `pytest`

```bash
# Only required once:
conda install -n srws_trial_run -c conda-forge -y pytest
# or
mamba install -n srws_trial_run -c conda-forge -y pytest

conda activate srws_trial_run
pytest
# or
python -m pytest
```

## Building and pushing the container

To build the container, use the script [build-container.bat](build-container.bat) on Windows, or
[build-container.sh](build-container.sh) on Linux.

The [Dockerfile](container/Dockerfile) is laid out in a way that it uses the cache as efficiently as possible.
The environment is only rebuilt if [environment.yml](environment.yml) is changed.



To actually run the job on Kubernetes, you must push it first. For this to happen, issue:

```bash
docker push 10.93.107.160:5000/giyash/srws_trial_run:latest
```

## Running the job

The Kubernetes Job Manifest is [srws_trial_run.yaml](kube/srws_trial_run.yaml).

To run it once, issue:

```bash
kubectl apply -f kube/srws_trial_run.yaml
```

After a successful run, the job will remain in completed state for 300 seconds after
being removed from the cluster. During this time, you can get its logs by issuing:

```bash
kubectl logs job/srws-trial-run
```

If the container executes a long-running task and emits a lot of logs, add `-f` to the call.


If you did not change the job manifest, your job will run in the namespace `ashim`.
If this is not your current active namespace, you must either:

 - prepend `-n` to each kubectl call: `kubectl -n ashim ...` or:
 - set the default namespace: `kubectl config set-context --current --namespace=ashim`


### Cleaning up

If your job is complete and you verified everything is OK, or if you want to cancel it, issue:

```bash
kubectl delete job/srws-trial-run
```

This will automatically be done after 300 seconds after completion.

### Subsequent runs

Kubernetes Jobs need to have unique names, so for each subsequent run, you *must* change the name. For this, edit the above file and change the name a bit.

### Parameters

If your Python script takes command-line arguments, you can add an `args:` section.

If your Python script uses environment variables, specify them in the optional `env:` section.