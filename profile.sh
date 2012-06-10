#! /usr/bin/env bash
# env CPUPROFILE=./a.prof HEAPPROFILE=./a.hprof CPUPROFILE_FREQUENCY=100 ./a.out $*
env CPUPROFILE=./a.prof CPUPROFILE_FREQUENCY=100 ./a.out $*
