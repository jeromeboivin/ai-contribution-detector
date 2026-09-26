# Credits

## Agent-commit dataset (`datasets/agent_commits/`)

The `*.jsonl.gz` files under `datasets/agent_commits/` contain excerpts of source code from the
51 open-source repositories listed below. All rights to that code remain with its authors.

**How the code was excerpted** (the statement of changes that the Apache-2.0, Zlib, CC-BY-4.0, GPL-3.0 and
AGPL-3.0 licenses ask for). On 2026-09-26, each row was made from one changed file of one commit: the lines
added by the commit plus up to 3 unchanged lines of context around each change, joined together, with removed
lines dropped and line endings normalized to `\n`. Nothing else was edited. Every row records its origin
(`repo`, `commit`, `path`) and its repository's license (`license`, an SPDX id), so the original file is at
`https://github.com/<repo>/blob/<commit>/<path>`.

**Licenses.** Each repository's license text — and, when the project has one, its NOTICE file — is reproduced
in a `LICENSES/` folder next to its rows, one file per repository. The license shown is the repository's license
at the time of the export.

- [`datasets/agent_commits/`](datasets/agent_commits/): permissive licenses (MIT, Apache-2.0, BSD-3-Clause, Zlib, CC-BY-4.0).
- [`datasets/agent_commits/copyleft/`](datasets/agent_commits/copyleft/): GPL-3.0 and AGPL-3.0, kept separate; those rows stay under their license — see its [NOTICE.md](datasets/agent_commits/copyleft/NOTICE.md).

**Selection.** The repositories come from the [qmmit agent-commit index](https://huggingface.co/datasets/balrampandey/qmmit-open-source-agent-commit-index) by Balram Pandey (qmmit
Dataset Terms: use and redistribution with attribution and a link to the source). Rows labelled `ai` come from
commits carrying a coding agent's own signature; rows labelled `human` from commits of the same repository
before June 2021. No contributor identities are included.

### Permissive licenses

| Repository | License | Human rows | AI rows | Split | License text |
|---|---|---:|---:|---|---|
| [ant-design/ant-design](https://github.com/ant-design/ant-design) | MIT | 150 | 150 | train | [LICENSES/ant-design__ant-design.txt](datasets/agent_commits/LICENSES/ant-design__ant-design.txt) |
| [ant-design/ant-design-pro](https://github.com/ant-design/ant-design-pro) | MIT | 132 | 132 | train | [LICENSES/ant-design__ant-design-pro.txt](datasets/agent_commits/LICENSES/ant-design__ant-design-pro.txt) |
| [apify/crawlee](https://github.com/apify/crawlee) | Apache-2.0 | 7 | 7 | train | [LICENSES/apify__crawlee.txt](datasets/agent_commits/LICENSES/apify__crawlee.txt) |
| [appsmithorg/appsmith](https://github.com/appsmithorg/appsmith) | Apache-2.0 | 135 | 135 | train | [LICENSES/appsmithorg__appsmith.txt](datasets/agent_commits/LICENSES/appsmithorg__appsmith.txt) |
| [BabylonJS/Babylon.js](https://github.com/BabylonJS/Babylon.js) | Apache-2.0 | 150 | 150 | validation | [LICENSES/BabylonJS__Babylon.js.txt](datasets/agent_commits/LICENSES/BabylonJS__Babylon.js.txt) |
| [backstage/backstage](https://github.com/backstage/backstage) | Apache-2.0 | 150 | 150 | train | [LICENSES/backstage__backstage.txt](datasets/agent_commits/LICENSES/backstage__backstage.txt) |
| [calcom/cal.diy](https://github.com/calcom/cal.diy) | MIT | 150 | 150 | train | [LICENSES/calcom__cal.diy.txt](datasets/agent_commits/LICENSES/calcom__cal.diy.txt) |
| [clauderic/dnd-kit](https://github.com/clauderic/dnd-kit) | MIT | 150 | 150 | train | [LICENSES/clauderic__dnd-kit.txt](datasets/agent_commits/LICENSES/clauderic__dnd-kit.txt) |
| [colinhacks/zod](https://github.com/colinhacks/zod) | MIT | 70 | 70 | train | [LICENSES/colinhacks__zod.txt](datasets/agent_commits/LICENSES/colinhacks__zod.txt) |
| [cypress-io/cypress](https://github.com/cypress-io/cypress) | MIT | 150 | 150 | train | [LICENSES/cypress-io__cypress.txt](datasets/agent_commits/LICENSES/cypress-io__cypress.txt) |
| [desktop/desktop](https://github.com/desktop/desktop) | MIT | 150 | 150 | train | [LICENSES/desktop__desktop.txt](datasets/agent_commits/LICENSES/desktop__desktop.txt) |
| [eggjs/egg](https://github.com/eggjs/egg) | MIT | 28 | 28 | train | [LICENSES/eggjs__egg.txt](datasets/agent_commits/LICENSES/eggjs__egg.txt) |
| [emberjs/ember.js](https://github.com/emberjs/ember.js) | MIT | 113 | 113 | test | [LICENSES/emberjs__ember.js.txt](datasets/agent_commits/LICENSES/emberjs__ember.js.txt) |
| [expo/expo](https://github.com/expo/expo) | MIT | 150 | 150 | test | [LICENSES/expo__expo.txt](datasets/agent_commits/LICENSES/expo__expo.txt) |
| [freeCodeCamp/freeCodeCamp](https://github.com/freeCodeCamp/freeCodeCamp) | BSD-3-Clause | 7 | 7 | train | [LICENSES/freeCodeCamp__freeCodeCamp.txt](datasets/agent_commits/LICENSES/freeCodeCamp__freeCodeCamp.txt) |
| [github/docs](https://github.com/github/docs) | CC-BY-4.0 | 80 | 80 | validation | [LICENSES/github__docs.txt](datasets/agent_commits/LICENSES/github__docs.txt) |
| [heroui-inc/heroui](https://github.com/heroui-inc/heroui) | Apache-2.0 | 136 | 136 | train | [LICENSES/heroui-inc__heroui.txt](datasets/agent_commits/LICENSES/heroui-inc__heroui.txt) |
| [homebridge/homebridge](https://github.com/homebridge/homebridge) | Apache-2.0 | 150 | 150 | train | [LICENSES/homebridge__homebridge.txt](datasets/agent_commits/LICENSES/homebridge__homebridge.txt) |
| [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) | Apache-2.0 | 8 | 8 | train | [LICENSES/HumanSignal__label-studio.txt](datasets/agent_commits/LICENSES/HumanSignal__label-studio.txt) |
| [Kong/insomnia](https://github.com/Kong/insomnia) | Apache-2.0 | 88 | 88 | train | [LICENSES/Kong__insomnia.txt](datasets/agent_commits/LICENSES/Kong__insomnia.txt) |
| [mantinedev/mantine](https://github.com/mantinedev/mantine) | MIT | 103 | 103 | test | [LICENSES/mantinedev__mantine.txt](datasets/agent_commits/LICENSES/mantinedev__mantine.txt) |
| [microsoft/playwright](https://github.com/microsoft/playwright) | Apache-2.0 | 91 | 91 | test | [LICENSES/microsoft__playwright.txt](datasets/agent_commits/LICENSES/microsoft__playwright.txt) |
| [microsoft/vscode](https://github.com/microsoft/vscode) | MIT | 150 | 150 | train | [LICENSES/microsoft__vscode.txt](datasets/agent_commits/LICENSES/microsoft__vscode.txt) |
| [motiondivision/motion](https://github.com/motiondivision/motion) | MIT | 150 | 150 | train | [LICENSES/motiondivision__motion.txt](datasets/agent_commits/LICENSES/motiondivision__motion.txt) |
| [neoclide/coc.nvim](https://github.com/neoclide/coc.nvim) | MIT | 44 | 44 | train | [LICENSES/neoclide__coc.nvim.txt](datasets/agent_commits/LICENSES/neoclide__coc.nvim.txt) |
| [nestjs/nest](https://github.com/nestjs/nest) | MIT | 98 | 98 | train | [LICENSES/nestjs__nest.txt](datasets/agent_commits/LICENSES/nestjs__nest.txt) |
| [nrwl/nx](https://github.com/nrwl/nx) | MIT | 150 | 150 | train | [LICENSES/nrwl__nx.txt](datasets/agent_commits/LICENSES/nrwl__nx.txt) |
| [palantir/blueprint](https://github.com/palantir/blueprint) | Apache-2.0 | 56 | 56 | train | [LICENSES/palantir__blueprint.txt](datasets/agent_commits/LICENSES/palantir__blueprint.txt) |
| [payloadcms/payload](https://github.com/payloadcms/payload) | MIT | 150 | 150 | train | [LICENSES/payloadcms__payload.txt](datasets/agent_commits/LICENSES/payloadcms__payload.txt) |
| [pixijs/pixijs](https://github.com/pixijs/pixijs) | MIT | 150 | 150 | train | [LICENSES/pixijs__pixijs.txt](datasets/agent_commits/LICENSES/pixijs__pixijs.txt) |
| [portainer/portainer](https://github.com/portainer/portainer) | Zlib | 5 | 5 | train | [LICENSES/portainer__portainer.txt](datasets/agent_commits/LICENSES/portainer__portainer.txt) |
| [QwikDev/qwik](https://github.com/QwikDev/qwik) | MIT | 150 | 150 | test | [LICENSES/QwikDev__qwik.txt](datasets/agent_commits/LICENSES/QwikDev__qwik.txt) |
| [react-hook-form/react-hook-form](https://github.com/react-hook-form/react-hook-form) | MIT | 82 | 82 | test | [LICENSES/react-hook-form__react-hook-form.txt](datasets/agent_commits/LICENSES/react-hook-form__react-hook-form.txt) |
| [recharts/recharts](https://github.com/recharts/recharts) | MIT | 150 | 150 | train | [LICENSES/recharts__recharts.txt](datasets/agent_commits/LICENSES/recharts__recharts.txt) |
| [remix-run/react-router](https://github.com/remix-run/react-router) | MIT | 48 | 48 | test | [LICENSES/remix-run__react-router.txt](datasets/agent_commits/LICENSES/remix-run__react-router.txt) |
| [remix-run/remix](https://github.com/remix-run/remix) | MIT | 88 | 88 | train | [LICENSES/remix-run__remix.txt](datasets/agent_commits/LICENSES/remix-run__remix.txt) |
| [solidjs/solid](https://github.com/solidjs/solid) | MIT | 32 | 32 | train | [LICENSES/solidjs__solid.txt](datasets/agent_commits/LICENSES/solidjs__solid.txt) |
| [storybookjs/storybook](https://github.com/storybookjs/storybook) | MIT | 150 | 150 | train | [LICENSES/storybookjs__storybook.txt](datasets/agent_commits/LICENSES/storybookjs__storybook.txt) |
| [styled-components/styled-components](https://github.com/styled-components/styled-components) | MIT | 53 | 53 | train | [LICENSES/styled-components__styled-components.txt](datasets/agent_commits/LICENSES/styled-components__styled-components.txt) |
| [supabase/supabase](https://github.com/supabase/supabase) | Apache-2.0 | 150 | 150 | test | [LICENSES/supabase__supabase.txt](datasets/agent_commits/LICENSES/supabase__supabase.txt) |
| [super-productivity/super-productivity](https://github.com/super-productivity/super-productivity) | MIT | 150 | 150 | train | [LICENSES/super-productivity__super-productivity.txt](datasets/agent_commits/LICENSES/super-productivity__super-productivity.txt) |
| [transloadit/uppy](https://github.com/transloadit/uppy) | MIT | 44 | 44 | train | [LICENSES/transloadit__uppy.txt](datasets/agent_commits/LICENSES/transloadit__uppy.txt) |
| [trpc/trpc](https://github.com/trpc/trpc) | MIT | 58 | 58 | train | [LICENSES/trpc__trpc.txt](datasets/agent_commits/LICENSES/trpc__trpc.txt) |
| [ueberdosis/tiptap](https://github.com/ueberdosis/tiptap) | MIT | 150 | 150 | train | [LICENSES/ueberdosis__tiptap.txt](datasets/agent_commits/LICENSES/ueberdosis__tiptap.txt) |
| [vitejs/vite](https://github.com/vitejs/vite) | MIT | 81 | 81 | train | [LICENSES/vitejs__vite.txt](datasets/agent_commits/LICENSES/vitejs__vite.txt) |
| [vuejs/vitepress](https://github.com/vuejs/vitepress) | MIT | 150 | 150 | validation | [LICENSES/vuejs__vitepress.txt](datasets/agent_commits/LICENSES/vuejs__vitepress.txt) |
| [xtermjs/xterm.js](https://github.com/xtermjs/xterm.js) | MIT | 125 | 125 | train | [LICENSES/xtermjs__xterm.js.txt](datasets/agent_commits/LICENSES/xtermjs__xterm.js.txt) |

### Copyleft licenses

| Repository | License | Human rows | AI rows | Split | License text |
|---|---|---:|---:|---|---|
| [Foundry376/Mailspring](https://github.com/Foundry376/Mailspring) | GPL-3.0 | 150 | 150 | train | [copyleft/LICENSES/Foundry376__Mailspring.txt](datasets/agent_commits/copyleft/LICENSES/Foundry376__Mailspring.txt) |
| [grafana/grafana](https://github.com/grafana/grafana) | AGPL-3.0 | 150 | 150 | train | [copyleft/LICENSES/grafana__grafana.txt](datasets/agent_commits/copyleft/LICENSES/grafana__grafana.txt) |
| [renovatebot/renovate](https://github.com/renovatebot/renovate) | AGPL-3.0 | 150 | 150 | train | [copyleft/LICENSES/renovatebot__renovate.txt](datasets/agent_commits/copyleft/LICENSES/renovatebot__renovate.txt) |
| [TriliumNext/Trilium](https://github.com/TriliumNext/Trilium) | AGPL-3.0 | 26 | 26 | train | [copyleft/LICENSES/TriliumNext__Trilium.txt](datasets/agent_commits/copyleft/LICENSES/TriliumNext__Trilium.txt) |

### Used locally, not redistributed

These repositories have custom or source-available licenses that were not reviewed for redistribution; their
rows exist only in local builds (`aicontrib build-agent-commits`):

- [beekeeper-studio/beekeeper-studio](https://github.com/beekeeper-studio/beekeeper-studio)
- [DefinitelyTyped/DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped)
- [directus/directus](https://github.com/directus/directus)
- [elastic/kibana](https://github.com/elastic/kibana)
- [foambubble/foam](https://github.com/foambubble/foam)
- [lucide-icons/lucide](https://github.com/lucide-icons/lucide)
- [medusajs/medusa](https://github.com/medusajs/medusa)
- [microsoft/fluentui](https://github.com/microsoft/fluentui)
- [n8n-io/n8n](https://github.com/n8n-io/n8n)
- [nocobase/nocobase](https://github.com/nocobase/nocobase)
- [nocodb/nocodb](https://github.com/nocodb/nocodb)
- [remotion-dev/remotion](https://github.com/remotion-dev/remotion)
- [RocketChat/Rocket.Chat](https://github.com/RocketChat/Rocket.Chat)
- [SigNoz/signoz](https://github.com/SigNoz/signoz)
- [teambit/bit](https://github.com/teambit/bit)
- [tldraw/tldraw](https://github.com/tldraw/tldraw)

## Coding-agent signature rules (`src/aicontrib/data/agent_commits.py`)

The agent-signature and CI-bot rules are ported from [qmmit-cli](https://github.com/pandey019/qmmit-cli)
(`src/signatures.js`), used under the MIT License:

```text
MIT License

Copyright (c) 2026 qmmit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
