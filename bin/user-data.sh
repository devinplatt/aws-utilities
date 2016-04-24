#!/bin/sh
# Note that we need to run /home/ubuntu/anaconda2/bin/python
# and NOT /usr/bin/python, so that imports work.
PARALLEL=1  # Number of parallel processes to run.
REPOS="/home/ubuntu/repos/"
SCRIPT="/home/ubuntu/repos/aws_utilities/bin/get_jobs.py"
COMMAND="/home/ubuntu/anaconda2/bin/python /home/ubuntu/repos/aws_utilities/bin/extract_one.py"
#PARAMS="<working directory> <SQS queue> <AWS region> <command>"
PARAMS="/var/tmp platt-feature-extraction us-west-1 $COMMAND"
#yum update -y  # Does not seem to be working.
cd $REPOS
git clone https://github.com/devinplatt/aws-utilities.git
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
