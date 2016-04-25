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

from extraction_worker.lib.core import timeit, ensure_dirs_exist
from extraction_worker.lib import extract_features
from extraction_worker.lib import create_examples


def get_jobs(work_dir, sqs_queue_name, aws_region):
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
                if action in ['extract_features', 'create_examples']:
                    s3_bucket_name = job[1]
                    s3_input_key = job[2]
                    s3_output_key = job[3]
                    status = process(s3, s3_bucket_name, s3_input_key,
                                     s3_output_key, work_dir, action)
                    if (status):
                        print("Message processed correctly ...")
                        m.delete()
                        print("Message deleted")
                else:
                    print('Falied to find option for action: {}'.format(action))


def process(s3, s3_bucket_name, s3_input_key, s3_output_key, work_dir, action):
    s3Bucket = s3.get_bucket(s3_bucket_name)
    local_input_path = os.path.join(work_dir, s3_input_key)
    local_output_path = os.path.join(work_dir, s3_output_key)
    ensure_dirs_exist([os.path.dirname(local_input_path),
                       os.path.dirname(local_output_path)])
    print("Downloading %s from s3://%s/%s ..." % (local_input_path, s3_bucket_name, s3_input_key))
    key = s3Bucket.get_key(s3_input_key)
    key.get_contents_to_filename(local_input_path)
    if action == 'extract_features':
        success = extract_features.try_extract_one(local_input_path,
                                                   local_output_path)
    elif action == 'create_examples':
        success = create_examples.try_extract_one(local_input_path,
                                                  local_output_path)
    else:
        print('Falied to find function for action: {}'.format(action))
        success = False
    if not success:
        print('Falied to extract features for: {}'.format(local_input_path))
        return False
    print("Uploading %s to s3://%s/%s ..." % (local_output_path, s3_bucket_name, s3_output_key))
    key = Key(s3Bucket)
    key.key = s3_output_key
    key.set_contents_from_filename(local_output_path)
    # TODO: delete local mp3 file!
    return True


def signal_handler(signal, frame):
    print("Exiting...")
    exit(0)


def main():
    if len(argv) < 4:
        print("Usage: %s <working directory> <SQS queue> <AWS region>" % argv[0])
        exit(1)
    work_dir = argv[1]
    sqs_queue_name = argv[2]
    aws_region = argv[3]
    get_jobs(work_dir, sqs_queue_name, aws_region)

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)
    main()
