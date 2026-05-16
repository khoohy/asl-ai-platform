# Backend

## Phase 4A: Real model loading

- Model path: `backend/models/asl_wlasl300_realtime.pt`
- Label map path: `backend/artifacts/label_map_300.json`
- Smoke test command:

```powershell
python backend\scripts\test_model_loading.py
```

- This phase only verifies that the production temporal model and label map can be loaded safely.
- Frame preprocessing, real image decoding, and real inference routes are intentionally deferred to Phase 4B and Phase 4C.
