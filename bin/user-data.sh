#!/bin/sh
# Note that we need to run /home/ubuntu/anaconda2/bin/python
# and NOT /usr/bin/python, so that imports work.
PARALLEL=1  # Number of parallel processes to run.
REPOS="/home/ubuntu/repos/"
SCRIPT="/home/ubuntu/repos/extraction_worker/bin/get_jobs.py"
#PARAMS="<working directory> <SQS queue> <AWS region>"
PARAMS="/var/tmp platt-feature-extraction us-west-1"
#yum update -y  # Does not seem to be working.
cd $REPOS
git clone https://github.com/devinplatt/extraction_worker.git
cd /
export PYTHONPATH=$REPOS:$PYTHONPATH
for i in $(seq $PARALLEL)
do
    LOGFILE=${SCRIPT}.$i.log
    echo "Starting $i of $PARALLEL - log file is $LOGFILE ..."
    DEBUGFILE=${SCRIPT}.$i.debug.log
    echo "The pwd is `pwd`" > $DEBUGFILE
    echo "The default python version run is `which python`" >> $DEBUGFILE
    nohup /home/ubuntu/anaconda2/bin/python $SCRIPT $PARAMS > $LOGFILE 2>&1 &
done
