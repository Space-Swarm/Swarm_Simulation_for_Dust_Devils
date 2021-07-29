#!/bin/sh
#
#PBS -1 nodes=1:ppn=1, walltime = 2:00:00

module add languages/python-3.7.7-anaconda-2020.20-grpMAX

# Execute Code
jupyter nbconvert --to notebook --execute Testing.ipynb
