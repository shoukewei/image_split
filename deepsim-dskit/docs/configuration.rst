Configuration
=============

Experiment configs describe data loading, splitting, preprocessing, candidate
models, and output locations. JSON and YAML are both supported.

The canonical examples are:

* ``configs/advertising_baseline.json``
* ``configs/advertising_baseline.yaml``

Use ``data.path`` for local data or ``data.url`` for remote data. Reader options
belong under ``data.read_kwargs``.
