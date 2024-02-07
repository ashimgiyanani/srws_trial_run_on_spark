#!/bin/bash

echo -n Preparing Python environment ...\ 
mkdir -p /tmp/env
tar xzf /app/env.tgz -C /tmp/env/
echo -n Activating Python environment ... \ 
source /tmp/env/bin/activate
echo -n Running post-install tasks ...\ 
/tmp/env/bin/conda-unpack
echo Done. Now running the job.

cd /app/srws_trial_run
python main.py "$@"
