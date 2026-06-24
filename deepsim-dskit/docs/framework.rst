Framework Extensions
====================

``dskit.framework`` provides a small registry for custom pipeline steps. A step
can be a class with ``fit`` and ``transform`` methods, or a stateless DataFrame
function registered with ``register_function_step``.

Example:

.. code-block:: python

   import pandas as pd
   from dskit import register_function_step, PreprocessingPipeline

   def add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
       df["total_spend"] = df["TV"] + df["radio"] + df["newspaper"]
       return df

   register_function_step("total_spend", add_total_spend)

   pipeline = PreprocessingPipeline({
       "steps": [{"name": "total_spend"}],
       "scaling": {"columns": ["total_spend"], "method": "standard"},
   })
