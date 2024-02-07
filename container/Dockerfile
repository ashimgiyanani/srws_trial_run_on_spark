# Micromamba is a version of Mamba to quickly create Conda environments.
# The build-env stage takes the environment.yml, creates a Conda env and packs it up.
# Later, when running the job, the env is temporarily unpacked right after start,
# and the program is run from there.

FROM mambaorg/micromamba:1.5.6 AS build-env

# Needed to be able to run Python-based commands in the base env, e.g. conda-pack later
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# We'll ship packed conda envs to make the container smaller and keep everything together nicely
RUN micromamba install -c conda-forge -y conda-pack



# Build the environment
WORKDIR /build-env
COPY environment.yml .
RUN micromamba env create -n env -f environment.yml &&\
    conda-pack -p /opt/conda/envs/env -o env.tgz

# Now, the base stage. This one prepares everything we need to run the job.
# We use multi-stage builds because the preparations can use different environments,
# and each step, even if the files are deleted later, is copied to the image.
# This means, if you create an image, add 5 GiB to one step, and delete them in the
# next step, the image will still be 5 GiB in size!

# The base runs on Debian bookworm. This is a bit larger than Alpine, but Alpine is missing
# some packages required for flawless function of Conda-based environments.

FROM debian:bookworm-slim AS base

WORKDIR /app

# Windows uses CRLF line endings. A Bash script can get confused by this. So we need an
# extra stage that copies the entrypoint and corrects line endings.

FROM base AS entrypoint

# Install dos2unix with APT. No need to clean the cache here because
# we'll later only copy over the entrypoint script.
RUN apt-get update && apt-get -y install dos2unix

COPY container/docker-entrypoint.sh .
RUN dos2unix docker-entrypoint.sh

# The final stage. This will house the ready-made application.
FROM base AS final

# Copy the entrypoint
COPY --chmod=777 --from=entrypoint /app/docker-entrypoint.sh .
# Copy the compressed env
COPY --from=build-env /build-env/env.tgz .
# Copy the app
COPY srws_trial_run srws_trial_run

# Update the container's system to catch some potential security flaws.
# This is the last step to ensure the cache is not used.
RUN apt-get update &&\
    apt-get -y upgrade &&\
    apt-get -y clean autoremove &&\
    rm -rf /var/cache/apt /var/lib/apt

# Specify the entrypoint to start
ENTRYPOINT ["bash", "/app/docker-entrypoint.sh"]
