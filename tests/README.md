# Sequence-Space Jacobian Test Suite

This is a `pytest` test suite, containing unit tests for the Sequence-Space Jacobian package.
Contained in the various directories in this test suite are major classes of tests:
- `base`: The unit tests for the base functionality of the package. These should be run after every commit
to the codebase.
- `robustness`: The unit tests to ensure the utilization of various features of the package are robust to bad initial
conditions or invalid user input. Some of these tests take a substantially longer time to execute and hence should be
run subject to user discretion.
- `performance`: The tests to gauge the performance (time-/allocation-wise) of the package to ensure no new bottlenecks
are created and that code performance remains in line with previous iterations of the package. These tests can also be
used for code optimization.