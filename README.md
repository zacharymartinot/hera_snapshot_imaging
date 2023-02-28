# HERA Snapshot Imaging

A simple tool for producing quick snapshot images from HERA data.

It does not flag, or calibrate, or check anything. There are almost no parameters to set, what parameters there are have overall minor effects, and should probably be removed in the future. You provide calibrated and flagged data, it makes snapshot images. Useful for checking that a calibration is at least reasonable - or horribly wrong!

Somewhat experimental. Outputs are not properly normalized yet. Coordinate orientation
is also screwed up, but the provided plotting function is internally consistent.

Requires `numpy`, `numba`, `scipy`, `matplotlib`, `astropy`.

Install with `pip install .`
