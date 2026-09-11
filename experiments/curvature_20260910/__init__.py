"""Frequency allocation as a constrained optimization, solved rather than searched.

Read README.md in this directory first.  The short version:

  tables.py       the m-coordinate, and the named families in it
  preflight.py    stage 0: free checks that catch a wrong algebra or a drifted env
  local_probe.py  the constraint side: output-KL and the Fisher it implies
  long_grad.py    the objective side: d(long loss)/d(log freq), forward only
  solve_kkt.py    the closed-form step, and the explain-vs-beat judge
  forward_check.py real forwards on the step, and the pre-registered gate
  panel_jobs.py   queue the solved table on the existing frozen-weight harness
  driver.sh       staged runner with the gates wired in
"""
