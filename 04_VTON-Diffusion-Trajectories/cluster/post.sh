#!/bin/bash
# DAGMan POST script for trajectory nodes: <job exit code> <retry number> <max retries>
# Report failure while retries are left (so DAGMan resubmits the run), but
# report success once they're used up, so one run that keeps failing doesn't
# stop the analysis node from running over everything that did finish.
ret="$1"; retry="$2"; max="$3"
if [ "$ret" != "0" ] && [ "$retry" -lt "$max" ]; then
  exit 1
fi
exit 0
