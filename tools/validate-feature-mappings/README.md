# validate-feature-mappings

A small TypeScript CLI that validates `data/models/feature_mappings.json` (the mapping artifact `app/ml/train_risk_model.py` writes) against a strict schema (zod) -- every presenter/artist cancellation rate must be a real probability in `[0, 1]`, and the required top-level fields must all be present. Guards against a malformed or truncated mappings file silently breaking the Streamlit dashboard's inference path.

## Usage

```bash
npm install
npm run build
npm run validate                 # validates the real ../../data/models/feature_mappings.json
node dist/validate.js path/to/file.json  # or validate a specific file
```

## Testing

```bash
npm run build
npm test
```
