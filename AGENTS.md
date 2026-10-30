# Working here with an agent

Before writing or changing `tlc` code, read the 3LC primer for coding agents:
<https://docs.3lc.ai/3lc/latest/llms.txt>. It holds the concepts that survive releases (Tables, Runs, metrics,
the anti-patterns) and says where to look for exact names. For anything about the 3LC Hub (deployments, plugins,
GPU nodes, Connections) read <https://docs.3lc.ai/3lc/latest/llms-hub.txt>.

Both are primers, not references: confirm symbols against the installed package (`help(tlc.X)`), and read the
documentation page a primer points at, served as markdown under `https://docs.3lc.ai/3lc/latest/_sources/<page>.md.txt`.

Repository specifics (layout, notebook metadata, the `tlc_tools` CLI, how to run the tests) are in
[CLAUDE.md](CLAUDE.md).
