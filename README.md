# NSI mean reversion

## How to generate Jupiter notebook

First, create the notebook with empty outputs.

    jupytext --to notebook analysis.py

Then, either run the notebook normally in Jupyter, or do:

    jupyter nbconvert --to ipynb --inplace --execute --allow-errors analysis.ipynb

To do the above all in one command:

    jupytext --to ipynb --pipe-fmt ipynb \
      --pipe 'jupyter nbconvert --to ipynb --execute --allow-errors --stdin --stdout' \
      analysis.py
