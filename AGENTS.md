# Working here with an agent

Before writing or changing `tlc` code, read 3LC's documents for coding agents: <https://docs.3lc.ai/llms-full.txt>
holds the primer for the Python package (Tables, Runs, metrics, the anti-patterns) and the primer for the 3LC Hub
(deployments, plugins, GPU nodes, Connections), with the pages they rest on; <https://docs.3lc.ai/llms.txt> is the
index.

The primers name concepts, not signatures: confirm symbols against the installed package (`help(tlc.X)`), and read
any documentation page as markdown by replacing `.html` with `.md` in its address.

Repository specifics (layout, notebook metadata, the `tlc_tools` CLI, how to run the tests) are in
[CLAUDE.md](CLAUDE.md).
