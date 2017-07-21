#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from pymongo import MongoClient
from time import sleep
from train_lstm import ex

client = MongoClient()
db = client.sacred
running_experiments = []


def start_experiment(config):
    run = ex.run(config_updates=config)
    try:
        db_entry = db.runs.find({'config': run.config})[0]
        running_experiments.append(db_entry['_id'])
    except IndexError:
        print("ERROR: Newly created experiment not found.")


def check_for_work():
    for _id in running_experiments:
        try:
            if db.runs.find({'_id': _id})[0]['status'] != 'RUNNING':
                running_experiments.remove(_id)
        except IndexError:
            running_experiments.remove(_id)
    if len(running_experiments) > 0:
        return None
    try:
        queued_run = db.runs.find({'status': 'QUEUED'})[0]
    except IndexError:
        return None
    print("Starting an experiment with the following configuration:")
    print(queued_run['config'])
    start_experiment(queued_run['config'])
    db.runs.delete_one({'_id': queued_run['_id']})


if __name__ == "__main__":
    while True:
        check_for_work()
        sleep(10)
