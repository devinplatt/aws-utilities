#!/usr/bin/env python

import sys

import extract_features

def main():
    input_mp3_file_name = sys.argv[1]
    output_feature_file_name = sys.argv[2]
    try:
        extract_features.extract_one(input_mp3_file_name,
                                     output_feature_file_name)
    except Exception as e:
        print(e)
        return False
    else:
        return True

if __name__ == '__main__':
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)