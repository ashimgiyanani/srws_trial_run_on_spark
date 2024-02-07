@echo off


docker build -t 10.93.107.160:5000/giyash/srws_trial_run:latest -f container\Dockerfile .


echo Before running the job, push 10.93.107.160:5000/giyash/srws_trial_run:latest to the registry.
echo To do this, issue:
echo docker push 10.93.107.160:5000/giyash/srws_trial_run:latest
