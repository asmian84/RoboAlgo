"""
lab_engine — isolated experiment sandbox.

Nothing in here is wired into the live scanner or confidence pipeline.
Modules are hired into production only after their scorecard earns it,
and retired by deleting the module plus its rows in lab_experiments.db.
"""
