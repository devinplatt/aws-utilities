#!/usr/bin/env python

import json
import os
import subprocess
import signal

from sys import argv, exit

import boto
import boto.s3
import boto.sqs

from boto.s3.key import Key
from boto.sqs.message import Message

from aws_utilities.utils.core_utils import timeit, ensure_dirs_exist


def get_jobs(work_dir, sqs_queue_name, aws_region, command):
    s3 = boto.s3.connect_to_region(aws_region)
    sqs = boto.sqs.connect_to_region(aws_region)
    sqs_queue =  sqs.lookup(sqs_queue_name)
    while (True):
        print("Getting messages from SQS queue...")
        messages = sqs_queue.get_messages(wait_time_seconds=20)
        if messages:
            for m in messages:
                print(m.get_body())
                job = json.loads(m.get_body())
                print("Message received: '%s'" % job)
                action = job[0]
                if action == 'process':
                    s3_bucket_name = job[1]
                    s3_input_key = job[2]
                    s3_output_key = job[3]
                    status = process(s3, s3_bucket_name, s3_input_key,
                                     s3_output_key, work_dir, command)
                    if (status):
                        print("Message processed correctly ...")
                        m.delete()
                        print("Message deleted")
                
def process(s3, s3_bucket_name, s3_input_key, s3_output_key, work_dir, command):
    s3Bucket = s3.get_bucket(s3_bucket_name)
    local_input_path = os.path.join(work_dir, s3_input_key)
    local_output_path = os.path.join(work_dir, s3_output_key)
    ensure_dirs_exist([os.path.dirname(local_input_path),
                       os.path.dirname(local_output_path)])
    print("Downloading %s from s3://%s/%s ..." % (local_input_path, s3_bucket_name, s3_input_key))
    key = s3Bucket.get_key(s3_input_key)
    key.get_contents_to_filename(local_input_path)
    full_command = [command, local_input_path, local_output_path]
    print("Executing: %s" % ' '.join(full_command))
    returncode = subprocess.call(full_command)
    if returncode != 0:
        print("Return Code not '0'!")
        return False
    print("Uploading %s to s3://%s/%s ..." % (local_output_path, s3_bucket_name, s3_output_key))
    key = Key(s3Bucket)
    key.key = s3_output_key
    key.set_contents_from_filename(local_output_path)
    return True

def signal_handler(signal, frame):
    print("Exiting...")
    exit(0)

def main():
    if len(argv) < 4:
        print("Usage: %s <working directory> <SQS queue> <AWS region> <command>" % argv[0])
        exit(1)
    work_dir = argv[1]
    sqs_queue_name = argv[2]
    aws_region = argv[3]
    command = argv[4]
    get_jobs(work_dir, sqs_queue_name, aws_region, command)

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)
    main()