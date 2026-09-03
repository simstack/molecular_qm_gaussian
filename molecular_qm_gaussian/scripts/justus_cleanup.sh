
echo " "
echo "### Final cleanup: Remove TMP_WORK_DIR ..."
echo " "
cd "${TMP_BASE_DIR}"
rm -rvf "${TMP_WORK_DIR}"
echo "END_TIME               = `date +'%y-%m-%d %H:%M:%S %s'`"
echo " "
echo "### Exiting with exit code '$exit_code' ..."
echo " "
exit $exit_code
"""
