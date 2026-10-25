# Pipeline audit

## Confirmed issues fixed

- Training generated engineered columns, but the API only applied the saved
  scaler before prediction.
- `Amount.rank()` and `Time.max()` made a transaction's features depend on
  unrelated rows in the same batch.
- The API did not require the feature engineer or validate the saved feature
  schema, so incompatible artifacts could appear ready.
- Missing model artifacts were reported as a generic prediction failure and the
  health endpoint did not distinguish application availability from model
  readiness.
- The README contained a wrong clone URL, stale benchmark claims, and an
  inaccurate description of the saved artifacts.

## Remaining considerations

- The current evaluation remains a stratified offline split. A chronological
  comparison would require confirming that `Time` represents an ordering that
  matches the intended deployment population.
- Docker build and Render deployment require the external Docker/hosting
  services and were not run in this local validation.
- Existing model files were produced before the new feature artifact and must
  be regenerated with `python train.py` before the API becomes ready.
