# conformal-prediction-mlb
Replication materials for Conformal Prediction: An Assumption-Free Approach to Quantifying Uncertainty in Baseball Forecasts

## Dependencies
Install all dependencies in a virtual environment with:
```
pip install -r requirements.txt
```

## Replicate
Replicate the analysis with:
```
python src/conformal.py
```
All outputs will be in the `figures/` directory.

## Notes
- **System**: MacOS; **Python version**: 3.11.8
- `Table 1` is an html table saved as a png. To do this I used the `selenium` module which, by default,
requires Chrome to be installed. It can also be modified to use Firefox, Safari, or Edge.
