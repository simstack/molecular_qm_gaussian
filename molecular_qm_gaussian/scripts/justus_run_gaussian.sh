
echo " "
echo "### Loading software module:"
echo " "
module unload chem/gaussian
module load chem/gaussian/g16.C.01
if test -z "$GAUSSIAN_VERSION"; then
  echo "ERROR: Failed to load module 'chem/gaussian/g16.C.01'."
  exit 101
fi
echo "GAUSSIAN_VERSION       = $GAUSSIAN_VERSION"
export GAUSS_SCRDIR="${TMP_WORK_DIR}"
echo "GAUSS_SCRDIR           = $GAUSS_SCRDIR"
echo "SLURM_SUBMIT_DIR       = $SLURM_SUBMIT_DIR"

echo " "
echo "### Copying input files to TMP_WORK_DIR:"
echo " "
GAUSSIAN_BASE="gaussian"
GAUSSIAN_COM_FILE="${GAUSSIAN_BASE}.com"
GAUSSIAN_LOG_FILE="${GAUSSIAN_BASE}.log"
GAUSSIAN_CHK_FILE="${GAUSSIAN_BASE}.chk"

cp -v  "$JOB_DIR/${GAUSSIAN_COM_FILE}" "${TMP_WORK_DIR}/"
echo "### Listing files in ${TMP_WORK_DIR}:"
ls -al "${TMP_WORK_DIR}/"  # List files in TMP_WORK_DIR to verify input files

echo " "
echo "### Running application ..."
echo " "
time g16 < "$GAUSSIAN_COM_FILE" > "$GAUSSIAN_LOG_FILE" 2>&1
exit_code=$?
echo ""
echo "Executable finished with exit code $exit_code"
echo " "

echo "### directory status"
ls -al "${TMP_WORK_DIR}/"

echo "### Copying  results to JOB DIR $JOB_DIR ..."
echo " "
cp "${TMP_WORK_DIR}/${GAUSSIAN_LOG_FILE}" "${JOB_DIR}/" || { echo "ERROR: Failed to copy log-file '${GAUSSIAN_LOG_FILE}' to submit directory '${JOB_DIR}'"; exit 103; }
cp "${TMP_WORK_DIR}/${GAUSSIAN_CHK_FILE}" "${JOB_DIR}/" || { echo "ERROR: Failed to copy chk-file '${GAUSSIAN_CHK_FILE}' to submit directory '${JOB_DIR}'"; exit 104; }


