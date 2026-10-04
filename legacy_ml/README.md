# Legacy ML Notebooks

Note: These notebooks contain a known bug related to `CancerHistory` preprocessing. 
The old cleaning process applied IQR outlier capping to all numeric columns.
Since `CancerHistory` is a binary column with Q1=0 and Q3=0, all 1s were capped to 0.
This resulted in `CancerHistory` being 0 for all rows.
Do not use these notebooks for model training without fixing this bug first.
