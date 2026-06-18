CityFlow assets can be mounted here.

Expected files for native simulation:
- `roadnet.json`
- `flow.json`
- `config.json`

When the `cityflow` Python package is unavailable or these files are absent, the simulator automatically uses the graph-based fallback in `scripts/06_barricade_simulator.py`.
