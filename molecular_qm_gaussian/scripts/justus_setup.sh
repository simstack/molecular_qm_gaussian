#!/bin/bash
echo " "
echo "### Setting up shell environment and defaults for environment vars ..."
echo " "
# Reset all language and locale dependencies (write floats with a dot "."):
unset LANG; export LC_ALL="C"
# Disable all external multi-threading => %NProcShared of g16 is in control
export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1
# Define fallbacks and "sanitize" important environment variables:
export USER="${USER:=`logname`}"
export SLURM_JOB_ID="${SLURM_JOB_ID:=`date +%s`}"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:=`pwd`}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:=`basename "$0"`}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME//[^a-zA-Z0-9._-]/_}"
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:=1}"
export SLURM_CPUS_ON_NODE="${SLURM_CPUS_ON_NODE:=1}"
export SLURM_NTASKS="${SLURM_NTASKS:=1}"
export JOB_DIR=$PWD
echo "JOB_DIR                = $JOB_DIR"
# Increase stack limit to 200M per worker (10M system default not sufficient):
ulimit -s 200000
#
#echo " "
#echo "### Printing basic job infos to stdout ..."
#echo " "
#echo "START_TIME             = `date +'%y-%m-%d %H:%M:%S %s'`"
#echo "HOSTNAME               = ${HOSTNAME}"
#echo "USER                   = ${USER}"
#echo "SLURM_JOB_NAME         = ${SLURM_JOB_NAME}"
#echo "SLURM_JOB_ID           = ${SLURM_JOB_ID}"
#echo "SLURM_SUBMIT_DIR       = ${SLURM_SUBMIT_DIR}"
#echo "SLURM_JOB_NUM_NODES    = ${SLURM_JOB_NUM_NODES}"
#echo "SLURM_CPUS_ON_NODE     = ${SLURM_CPUS_ON_NODE}"
#echo "SLURM_NTASKS           = ${SLURM_NTASKS}"
#echo "SLURM_JOB_NODELIST     = ${SLURM_JOB_NODELIST}"
#echo "---------------- ulimit -a -S ----------------"
ulimit -a -S
#echo "---------------- ulimit -a -H ----------------"
ulimit -a -H
#echo "----------------------------------------------"

echo " "
echo "### Creating TMP_WORK_DIR directory and changing to it ..."
echo " "
# Using "${SCRATCH}" is recommended since Gaussian jobs can create a lot of disk IO:
# NEVER EVER calculate in your home directory.
if test -n "${SCRATCH}" -a -e "${SCRATCH}" -a -d "${SCRATCH}" -a "${SCRATCH}" != "/scratch" -a "${SCRATCH}" != "/tmp" -a "${SCRATCH}" != "/ramdisk"; then
  TMP_BASE_DIR="${SCRATCH:=/tmp/${USER}}"
else
  TMP_BASE_DIR="${TMPDIR:=/tmp/${USER}}"
fi
JOB_WORK_DIR="${SLURM_JOB_NAME}.${SLURM_JOB_ID%%.*}"
TMP_WORK_DIR="${TMP_BASE_DIR}/${JOB_WORK_DIR}"
echo "TMP_BASE_DIR           = ${TMP_BASE_DIR}"
echo "JOB_WORK_DIR           = ${JOB_WORK_DIR}"
echo "TMP_WORK_DIR           = ${TMP_WORK_DIR}"
mkdir -vp "${TMP_WORK_DIR}"
cd "${TMP_WORK_DIR}"
echo "in temp dir"









