# Experimental configurations

Hydra presets used only for rendering and physics comparisons live here. They
are deliberately outside the production `config/` tree and are loaded through
`experiments/main.py`:

```bash
python experiments/main.py \
  --config-dir experiments/config \
  --config-name render/level3_pick_visual_gallery
```

The experimental runner adds the production `config/` directory to Hydra's
search path, so these presets may inherit a real benchmark task without copying
it.
