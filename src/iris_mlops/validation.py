"""
Data validation schemas.

WHY PANDERA:
    ML fails silently. A model trained on data with swapped columns will
    report high accuracy and be useless in production. Pandera makes
    schemas explicit and raises when data violates them.
"""

from pandera.pandas import Check, Column, DataFrameSchema

# WHY range checks:
#   Iris measurements are always between 0 and 10 cm. A value of -5 or 999
#   indicates a data pipeline bug. Catch it BEFORE training.
#
# WHY checks are passed as a list:
#   Column's second parameter is `checks`, which expects a list. Passing
#   multiple checks as separate positional arguments makes Pandera treat
#   the second one as `nullable`, which fails silently at schema definition
#   time and blows up at validation time.
iris_schema = DataFrameSchema({
    "f0": Column(float, [Check.gt(0), Check.lt(10)], name="sepal_length"),
    "f1": Column(float, [Check.gt(0), Check.lt(10)], name="sepal_width"),
    "f2": Column(float, [Check.gt(0), Check.lt(10)], name="petal_length"),
    "f3": Column(float, [Check.gt(0), Check.lt(10)], name="petal_width"),
    "target": Column(int, Check.isin([0, 1, 2])),
})