# molecular_qm_gaussian

Gaussian 16 node for the Simstack framework.

Install from this repository:

```bash
uv pip install .
```

The `gaussian` node writes `gaussian.com`, runs `g16` from
`[<resource>.program.gaussian]` in `config.toml`, and parses `gaussian.log`.
Gaussian itself is licensed and is not shipped with this package.
