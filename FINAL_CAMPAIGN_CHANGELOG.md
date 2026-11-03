# FINAL_CAMPAIGN_CHANGELOG

- Reorganized historical results under archive_previous/.
- Excluded .git and __pycache__ from the delivery package.
- Added --max-vehicles to shadai_sumo.py.
- Added max_vehicles metadata to SHADAI CSV outputs.
- Upgraded RNG_SCHEME from split_v1 to split_v2.
- Added isolated energy-consumption RNG stream.
- Removed hidden/global RNG use from veiculo.py.
- Removed malicious outcome re-sampling inside Veiculo.atualizar_reputacao().
- Updated all final runners/analyzers that checked split_v1 to split_v2.
- Updated revs_sumo.py to split_v2, injected reputation/energy RNG, and --max-vehicles.
- Added final scalability runner/analyzer for 100/400/800 x 30 seeds.
- Added same-seed reproducibility verifier.
- Added master campaign, analysis and plotting runners.
- Rebuilt plotting scripts with explicit input sources, bars/lines plus 95% CI,
  and preserved original Score/Utility/Reputation figure concepts.
- Aligned C22/C23 final strategy comparisons with C24 by explicitly enabling dynamic pricing (alpha=beta=0.20).
