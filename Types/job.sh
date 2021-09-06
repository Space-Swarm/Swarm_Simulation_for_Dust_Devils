#!/bin/sh
#
#PBS -l select=1:ncpus=1:mem=100G
#PBS -l walltime=02:00:00
#PBS -J 0-9

module add lang/python/anaconda/keras
module add lang/python/anaconda/pytorch
module add lang/python/anaconda/2.7-2019.03.bioconda
module add lang/python/anaconda/2.7-2019.03.biopython
module add lang/python/anaconda/2.7-2019.10
module add lang/python/anaconda/3.7-2019.03.bioconda
module add lang/python/anaconda/3.7-2019.03.biopython
module add lang/python/anaconda/3.7-2019.03.ldsc
module add lang/python/anaconda/3.7-2019.03-tensorflow
module add lang/python/anaconda/3.7-2019.03
module add lang/python/anaconda/3.7-2019.10-isaac-gym
module add lang/python/anaconda/3.7-2019.10
module add lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
module add lang/python/anaconda/3.7-2020.04-tensorflow-2.4.1
module add lang/python/anaconda/3.7.3-2019.03-tensorflow-2.0
module add lang/python/anaconda/3.7.7-2020-R-3.6.1
module add lang/python/anaconda/3.8-2020.07
module add lang/python/anaconda/3.8-2021-TW
module add lang/python/anaconda/3.8.3-2020-math
module add lang/python/anaconda/3.8.5-2020-stablebaselines3
module add lang/python/anaconda/3.8.5-2021-AM
module add lang/python/anaconda/3.8.5-2021-cuda-11.2.2
module add lang/python/anaconda/3.8.5-2021-Jupyter  


#Change into working directory
cd $PBS_O_WORKDIR

#Execute code
python 'run_static.py' ${PBS_ARRAY_INDEX}

sleep 60
