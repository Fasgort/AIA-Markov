#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os

import time
from Robot import Robot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

__date__ = '2017.02.15'

"""
T2 - Hidden Markov Model Robot
"""

_DEFAULT_OUTPUT_PATH = './Output/'


def main(args):
    logging.info("Hidden Markov Model of Robot on a map")
    logging.debug("Verbose mode enabled")

    demo()
    model_building_performance_test()
    algorithms_performance_test()


def demo():
    output_messages = []
    output_file = _DEFAULT_OUTPUT_PATH + 'demo.txt'
    logging.info("Demonstration of use")
    size_x = 20
    obstacle_rate = 0.5
    observation_error = 0.05
    sample_size = 15
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_obstacle_rate(obstacle_rate)
    r.set_error(observation_error)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()
    r.make_b_mat()
    sam_s, sam_o = r.generate_sample(sample_size)
    forward_estimated_state = r.forward(sam_o)
    forward_error = r.forward_error(sam_s[-1], forward_estimated_state)
    viterbi_estimate_seq = r.viterbi(sam_o)
    viterbi_error = r.viterbi_error(sam_s, viterbi_estimate_seq)
    r.display_map()
    output_messages.append("Map size:\t{}".format(size_x))
    output_messages.append("Obstacle rate:\t{}".format(obstacle_rate))
    output_messages.append("Observation error:\t{}".format(observation_error))
    output_messages.append("Sample size:\t{}".format(sample_size))
    output_messages.append("State sequence:\t{}".format(sam_s))
    output_messages.append("Observation sequence:\t{}".format(sam_o))
    output_messages.append("Forward estimate state:\t{}".format(forward_estimated_state))
    output_messages.append("Forward error:\t{}".format(forward_error))
    output_messages.append("Viterbi estimate sequence:\t{}".format(viterbi_estimate_seq))
    output_messages.append("Viterbi error:\t{}".format(viterbi_error))
    for msg in output_messages:
        logging.info(msg)
    try:
        fd = open(output_file, 'w')
        r.save_map_image(_DEFAULT_OUTPUT_PATH + 'demo_map.png')
        fd.write("Demonstration of use" + "\n")
        fd.write('\n'.join(output_messages) + '\n')
    except ValueError as ex:
        logging.warning("There was an error writing output file: {} \nException:\t{}".format(output_file, ex))


def model_building_performance_test():
    logging.info("Building model performance test")
    size_x = 15
    obstacle_rate = 0.5
    observation_error = 0.05

    performance_model_building_size = []
    performance_sample_generation = []

    logging.debug("Model size vs building time")
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("Map size\ttime(s)")
    for size_x in range(3, 18):
        ti = time.time()
        r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.set_error(observation_error)
        r.generate_map()
        r.make_a_mat()
        r.make_pi_v()
        r.make_b_mat()
        performance_model_building_size.append((size_x, time.time() - ti))
        logging.debug("({}, {})\t{}".format(size_x, size_x, performance_model_building_size[-1][1]))
    ax.plot(*zip(*performance_model_building_size),
            label='Model size / building time')
    ax.legend(loc='upper left')
    plt.xlabel('State sequence length')
    plt.ylabel('Elapsed time (s)')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Building model performance size.png')
    plt.clf()

    logging.debug("Sample size vs generation time")
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_error(observation_error)
    r.set_obstacle_rate(obstacle_rate)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()
    r.make_b_mat()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("State_sample({})\ttime(s)")
    for sample_size in range(1, 100):
        ti = time.time()
        r.generate_sample(sample_size)
        performance_sample_generation.append((sample_size, time.time() - ti))
        logging.debug("{}\t{}".format(sample_size, performance_sample_generation[-1][1]))
    ax.plot(*zip(*performance_sample_generation),
            label='Generation time')
    ax.legend(loc='upper left')
    plt.xlabel('State sequence length')
    plt.ylabel('Elapsed time (s)')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Building model performance sample size.png')
    plt.clf()


def algorithms_performance_test():
    logging.info("Algorithms performance test")
    size_x = 15
    obstacle_rate = 0.3
    observation_error = 0.05

    iterations = 2

    logging.debug("Forward time performance")
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_error(observation_error)
    r.set_obstacle_rate(obstacle_rate)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()
    r.make_b_mat()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("State_sample({})\ttime(s)")
    results = []
    for t in range(iterations):
        forward_performance_time_test = []
        for sample_size in range(2, 17):
            sam_s, sam_o = r.generate_sample(sample_size)
            ti = time.time()
            r.forward(sam_o)
            forward_performance_time_test.append(time.time() - ti)
            logging.debug("{0}\t{1:.2f}".format(sample_size, forward_performance_time_test[-1]))
        results.append(forward_performance_time_test)
    forward_performance_time_test = np.mean(results, axis=0)
    ax.plot(range(2, 17), forward_performance_time_test,
            label='Forward execution time')
    ax.legend(loc='upper left')
    plt.xlabel('State sequence length')
    plt.ylabel('Elapsed time (s)')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Time performance Forward.png')
    plt.clf()

    logging.debug("Viterbi time performance")
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_error(observation_error)
    r.set_obstacle_rate(obstacle_rate)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()
    r.make_b_mat()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("State_sample\ttime(s)")
    results = []
    for t in range(iterations):
        viterbi_performance_time_test = []
        for sample_size in range(2, 17):
            sam_s, sam_o = r.generate_sample(sample_size)
            ti = time.time()
            r.viterbi(sam_o)
            viterbi_performance_time_test.append(time.time() - ti)
            logging.debug("{0}\t{1:.2f}".format(sample_size, viterbi_performance_time_test[-1]))
        results.append(viterbi_performance_time_test)
    viterbi_performance_time_test = np.mean(results, axis=0)

    ax.plot(range(2, 17), viterbi_performance_time_test,
            label='Viterbi execution time')
    ax.legend(loc='upper left')
    plt.xlabel('State sequence length')
    plt.ylabel('Elapsed time (s)')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Time performance Viterbi.png')
    plt.clf()

    logging.debug("Forward performance")
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_error(observation_error)
    r.set_obstacle_rate(obstacle_rate)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()
    r.make_b_mat()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("State_sample\tdeviation")
    results = []
    for t in range(iterations):
        logging.debug("Iteration:\t{}".format(t))
        forward_performance_test = []
        for sample_size in range(10, 37):
            sam_s, sam_o = r.generate_sample(sample_size)
            forward_estimated_state = r.forward(sam_o)
            forward_performance_test.append(r.forward_error(sam_s[-1], forward_estimated_state))
            logging.debug("{}\t{}".format(sample_size, forward_performance_test[-1]))
        results.append(forward_performance_test)
    forward_performance_test = np.mean(results, axis=0)
    ax.plot(range(10, 37), forward_performance_test,
            label='Forward error')
    ax.legend(loc='upper left')
    plt.xlabel('State sequence length')
    plt.ylabel('Deviation')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Performance Forward.png')

    logging.debug("Viterbi performance")
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_error(observation_error)
    r.set_obstacle_rate(obstacle_rate)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()
    r.make_b_mat()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("State_sample\terror")
    results = []
    for t in range(iterations):
        logging.debug("Iteration:\t{}".format(t))
        viterbi_performance_test = []
        for sample_size in range(9, 37):
            sam_s, sam_o = r.generate_sample(sample_size)
            viterbi_estimate_seq = r.viterbi(sam_o)
            viterbi_performance_test.append(r.viterbi_error(sam_s, viterbi_estimate_seq))
            logging.debug("{}\t{}".format(sample_size, viterbi_performance_test[-1]))
        results.append(viterbi_performance_test)
    viterbi_performance_test = np.mean(results, axis=0)
    ax.plot(range(9, 37), viterbi_performance_test,
            label='Viterbi error')
    ax.legend(loc='upper left')
    plt.xlabel('State sequence length')
    plt.ylabel('Error')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Performance Viterbi.png')
    plt.clf()

    logging.debug("Forward performance vs observation error")
    sample_size = 15
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_obstacle_rate(obstacle_rate)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("Observation error\terror")
    results = []
    for t in range(iterations):
        logging.debug("Iteration:\t{}".format(t))
        forward_performance_observation_error_test = []
        for observation_error in np.arange(0, .3, 0.01):
            r.set_error(observation_error)
            r.make_b_mat()
            sam_s, sam_o = r.generate_sample(sample_size)
            forward_estimated_state = r.forward(sam_o)
            forward_performance_observation_error_test.append(r.forward_error(sam_s[-1], forward_estimated_state))
            logging.debug("{}\t{}".format(observation_error, forward_performance_observation_error_test[-1]))
        results.append(forward_performance_observation_error_test)
    forward_performance_observation_error_test = np.mean(results, axis=0)
    ax.plot(np.arange(0, .3, 0.01), forward_performance_observation_error_test,
            label='Forward error vs observation error')
    ax.legend(loc='upper left')
    plt.xlabel('Observation error')
    plt.ylabel('Deviation')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Performance Forward vs observation error.png')

    logging.debug("Viterbi performance vs observation error")
    sample_size = 15
    r = Robot(map_size=size_x, obstacle_rate=obstacle_rate, observation_error=observation_error)
    r.set_size(size_x)
    r.set_obstacle_rate(obstacle_rate)
    r.generate_map()
    r.make_a_mat()
    r.make_pi_v()
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    logging.debug("Observation error\terror")
    results = []
    for t in range(iterations):
        logging.debug("Iteration:\t{}".format(t))
        viterbi_performance_observation_error_test = []
        for observation_error in np.arange(0, .3, 0.01):
            r.set_error(observation_error)
            r.make_b_mat()
            sam_s, sam_o = r.generate_sample(sample_size)
            viterbi_estimate_seq = r.viterbi(sam_o)
            viterbi_performance_observation_error_test.append(r.viterbi_error(sam_s, viterbi_estimate_seq))
            logging.debug("{}\t{}".format(observation_error, viterbi_performance_observation_error_test[-1]))
        results.append(viterbi_performance_observation_error_test)
    viterbi_performance_observation_error_test = np.mean(results, axis=0)
    ax.plot(np.arange(0, .3, 0.01), viterbi_performance_observation_error_test,
            label='Viterbi error vs observation error')
    ax.legend(loc='upper left')
    plt.xlabel('Observation error')
    plt.ylabel('Error')
    fig.savefig(_DEFAULT_OUTPUT_PATH + 'Performance Viterbi vs observation error.png')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="AIA T2 - Hidden Markov Model")
    parser.add_argument(
        "-o",
        "--output",
        help="output data path",
        default=_DEFAULT_OUTPUT_PATH, type=str)
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Setup logging
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    main(args)
