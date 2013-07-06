#!/bin/sh

g++ -std=c++0x -pedantic -Wall -Wextra -O3 -o rg-train rg-train.cxx
g++ -std=c++0x -pedantic -Wall -Wextra -O3 -o rg-gearman-worker rg-gearman-worker.cxx -lgearman

