# Model Integration Plan

## Goal

Integrate the existing ASL recognition model into this platform without copying the entire legacy repository into `asl-ai-platform`.

## Phase-by-phase approach

### Phase 1

- scaffold the backend and frontend structure
- expose `/health`
- expose `/api/inference/mock`
- document the intended integration boundary

### Phase 2

- define a stable inference contract between platform and model
- add configuration for model location and versioning
- create a dedicated adapter module in the backend service layer
- standardize preprocessing expectations between platform and model code

### Phase 3

- package or export the model from the existing ASL repository
- load the model through a predictable runtime interface
- implement real inference in the backend
- add structured error handling, logging, and request tracing

### Phase 4

- benchmark latency and throughput
- containerize backend and frontend for deployment
- add CI validation and smoke tests
- introduce observability and operational safeguards

## Recommended integration options

### Option A: packaged model dependency

Publish a lightweight Python package from the model repository that exposes a clean inference API.

Pros:

- strongest separation between product and research code
- easier version pinning
- easier rollback path

Tradeoff:

- requires packaging discipline in the model repo

### Option B: exported model artifact

Export model weights and load them from this backend with shared preprocessing code.

Pros:

- simpler if the model is already stable
- keeps app repo smaller than a full repo copy

Tradeoff:

- inference preprocessing must be kept in sync manually

### Option C: external model service

Deploy the model separately and call it from this platform.

Pros:

- clear service boundary
- independent scaling

Tradeoff:

- higher operational complexity early on

## Suggested near-term direction

For this project, the best next step is usually Option A or Option B:

- choose Option A if you expect the model code to continue evolving and want a clean reusable interface
- choose Option B if the model architecture is already stable and you mainly need predictable inference assets

## Guardrails

- do not copy large model files into this repo
- do not mix training notebooks into the backend application package
- do not expose internal model objects directly through API routes
- keep request and response schemas stable even if the model internals change
