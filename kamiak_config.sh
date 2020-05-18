#
# Config file for running on high performance cluster with Slurm
# Note: paths for rsync, so make sure all paths have a trailing slash
#
modelFolder="kamiak-models"
logFolder="kamiak-logs"
remotessh="kamiak"  # in your .ssh/config file
project_name="$(basename "$(pwd)")"
remotedir="/data/vcea/garrett.wilson/${project_name}/"
localdir="/home/garrett/Documents/Github/${project_name}/"
