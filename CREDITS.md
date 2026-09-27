# Credits

## Agent-commit dataset (`datasets/agent_commits/`)

The `*.jsonl.gz` files under `datasets/agent_commits/` contain excerpts of source code from the
142 open-source repositories listed below. All rights to that code remain with its authors.

**How the code was excerpted** (the statement of changes that several of these licenses ask for, e.g.
Apache-2.0, the GPL, LGPL and MPL). On 2026-09-27, each row was made from one changed file of one commit: the lines
added by the commit plus up to 3 unchanged lines of context around each change, joined together, with removed
lines dropped and line endings normalized to `\n`. Nothing else was edited. Every row records its origin
(`repo`, `commit`, `path`) and its repository's license (`license`, an SPDX id), so the original file is at
`https://github.com/<repo>/blob/<commit>/<path>`.

**Licenses.** Each repository's license text — and, when the project has one, its NOTICE file — is reproduced
in a `LICENSES/` folder next to its rows, one file per repository. The license shown is the repository's license
at the time of the export.

- [`datasets/agent_commits/`](datasets/agent_commits/): permissive licenses (Apache-2.0, BSD-3-Clause, CC-BY-4.0, MIT, Zlib).
- [`datasets/agent_commits/copyleft/`](datasets/agent_commits/copyleft/): copyleft licenses (AGPL-3.0, GPL-2.0, GPL-3.0, LGPL-2.1, MPL-2.0), kept separate; those rows stay under their license — see its [NOTICE.md](datasets/agent_commits/copyleft/NOTICE.md).

**Selection.** The repositories come from the [qmmit agent-commit index](https://huggingface.co/datasets/balrampandey/qmmit-open-source-agent-commit-index) by Balram Pandey (qmmit
Dataset Terms: use and redistribution with attribution and a link to the source). Rows labelled `ai` come from
commits carrying a coding agent's own signature; rows labelled `human` from commits of the same repository
before June 2021. No contributor identities are included.

### Permissive licenses

| Repository | Language | License | Human rows | AI rows | Split | License text |
|---|---|---|---:|---:|---|---|
| [ardalis/CleanArchitecture](https://github.com/ardalis/CleanArchitecture) | C# | MIT | 9 | 9 | train | [LICENSES/ardalis__CleanArchitecture.txt](datasets/agent_commits/LICENSES/ardalis__CleanArchitecture.txt) |
| [AvaloniaUI/Avalonia](https://github.com/AvaloniaUI/Avalonia) | C# | MIT | 216 | 216 | train | [LICENSES/AvaloniaUI__Avalonia.txt](datasets/agent_commits/LICENSES/AvaloniaUI__Avalonia.txt) |
| [dotnet/aspnetcore](https://github.com/dotnet/aspnetcore) | C# | MIT | 244 | 244 | train | [LICENSES/dotnet__aspnetcore.txt](datasets/agent_commits/LICENSES/dotnet__aspnetcore.txt) |
| [dotnet/maui](https://github.com/dotnet/maui) | C# | MIT | 244 | 244 | validation | [LICENSES/dotnet__maui.txt](datasets/agent_commits/LICENSES/dotnet__maui.txt) |
| [dotnet/roslyn](https://github.com/dotnet/roslyn) | C# | MIT | 244 | 244 | train | [LICENSES/dotnet__roslyn.txt](datasets/agent_commits/LICENSES/dotnet__roslyn.txt) |
| [dotnet/runtime](https://github.com/dotnet/runtime) | C# | MIT | 244 | 244 | train | [LICENSES/dotnet__runtime.txt](datasets/agent_commits/LICENSES/dotnet__runtime.txt) |
| [icsharpcode/ILSpy](https://github.com/icsharpcode/ILSpy) | C# | MIT | 53 | 53 | train | [LICENSES/icsharpcode__ILSpy.txt](datasets/agent_commits/LICENSES/icsharpcode__ILSpy.txt) |
| [PowerShell/PowerShell](https://github.com/PowerShell/PowerShell) | C# | MIT | 6 | 6 | train | [LICENSES/PowerShell__PowerShell.txt](datasets/agent_commits/LICENSES/PowerShell__PowerShell.txt) |
| [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | C# | Apache-2.0 | 167 | 167 | train | [LICENSES/QuantConnect__Lean.txt](datasets/agent_commits/LICENSES/QuantConnect__Lean.txt) |
| [apache/arrow](https://github.com/apache/arrow) | C++ | Apache-2.0 | 48 | 48 | test | [LICENSES/apache__arrow.txt](datasets/agent_commits/LICENSES/apache__arrow.txt) |
| [apache/brpc](https://github.com/apache/brpc) | C++ | Apache-2.0 | 25 | 25 | validation | [LICENSES/apache__brpc.txt](datasets/agent_commits/LICENSES/apache__brpc.txt) |
| [dmlc/xgboost](https://github.com/dmlc/xgboost) | C++ | Apache-2.0 | 10 | 10 | validation | [LICENSES/dmlc__xgboost.txt](datasets/agent_commits/LICENSES/dmlc__xgboost.txt) |
| [duckdb/duckdb](https://github.com/duckdb/duckdb) | C++ | MIT | 110 | 110 | train | [LICENSES/duckdb__duckdb.txt](datasets/agent_commits/LICENSES/duckdb__duckdb.txt) |
| [electron/electron](https://github.com/electron/electron) | C++ | MIT | 110 | 110 | train | [LICENSES/electron__electron.txt](datasets/agent_commits/LICENSES/electron__electron.txt) |
| [envoyproxy/envoy](https://github.com/envoyproxy/envoy) | C++ | Apache-2.0 | 110 | 110 | train | [LICENSES/envoyproxy__envoy.txt](datasets/agent_commits/LICENSES/envoyproxy__envoy.txt) |
| [fmtlib/fmt](https://github.com/fmtlib/fmt) | C++ | MIT | 8 | 8 | test | [LICENSES/fmtlib__fmt.txt](datasets/agent_commits/LICENSES/fmtlib__fmt.txt) |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | C++ | MIT | 110 | 110 | train | [LICENSES/microsoft__onnxruntime.txt](datasets/agent_commits/LICENSES/microsoft__onnxruntime.txt) |
| [microsoft/winget-cli](https://github.com/microsoft/winget-cli) | C++ | MIT | 13 | 13 | train | [LICENSES/microsoft__winget-cli.txt](datasets/agent_commits/LICENSES/microsoft__winget-cli.txt) |
| [nlohmann/json](https://github.com/nlohmann/json) | C++ | MIT | 47 | 47 | train | [LICENSES/nlohmann__json.txt](datasets/agent_commits/LICENSES/nlohmann__json.txt) |
| [opencv/opencv](https://github.com/opencv/opencv) | C++ | Apache-2.0 | 10 | 10 | train | [LICENSES/opencv__opencv.txt](datasets/agent_commits/LICENSES/opencv__opencv.txt) |
| [ossrs/srs](https://github.com/ossrs/srs) | C++ | MIT | 53 | 53 | train | [LICENSES/ossrs__srs.txt](datasets/agent_commits/LICENSES/ossrs__srs.txt) |
| [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) | C++ | Apache-2.0 | 110 | 110 | train | [LICENSES/PaddlePaddle__Paddle.txt](datasets/agent_commits/LICENSES/PaddlePaddle__Paddle.txt) |
| [rui314/mold](https://github.com/rui314/mold) | C++ | MIT | 110 | 110 | train | [LICENSES/rui314__mold.txt](datasets/agent_commits/LICENSES/rui314__mold.txt) |
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | C++ | Apache-2.0 | 1 | 1 | validation | [LICENSES/tensorflow__tensorflow.txt](datasets/agent_commits/LICENSES/tensorflow__tensorflow.txt) |
| [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) | C++ | Apache-2.0 | 7 | 7 | train | [LICENSES/tesseract-ocr__tesseract.txt](datasets/agent_commits/LICENSES/tesseract-ocr__tesseract.txt) |
| [uNetworking/uWebSockets](https://github.com/uNetworking/uWebSockets) | C++ | Apache-2.0 | 8 | 8 | train | [LICENSES/uNetworking__uWebSockets.txt](datasets/agent_commits/LICENSES/uNetworking__uWebSockets.txt) |
| [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) | C++ | MIT | 15 | 15 | train | [LICENSES/yhirose__cpp-httplib.txt](datasets/agent_commits/LICENSES/yhirose__cpp-httplib.txt) |
| [Automattic/mongoose](https://github.com/Automattic/mongoose) | JavaScript | MIT | 12 | 12 | train | [LICENSES/Automattic__mongoose.txt](datasets/agent_commits/LICENSES/Automattic__mongoose.txt) |
| [jsdom/jsdom](https://github.com/jsdom/jsdom) | JavaScript | MIT | 133 | 133 | train | [LICENSES/jsdom__jsdom.txt](datasets/agent_commits/LICENSES/jsdom__jsdom.txt) |
| [krisk/Fuse](https://github.com/krisk/Fuse) | JavaScript | Apache-2.0 | 36 | 36 | validation | [LICENSES/krisk__Fuse.txt](datasets/agent_commits/LICENSES/krisk__Fuse.txt) |
| [mochajs/mocha](https://github.com/mochajs/mocha) | JavaScript | MIT | 38 | 38 | train | [LICENSES/mochajs__mocha.txt](datasets/agent_commits/LICENSES/mochajs__mocha.txt) |
| [mrdoob/three.js](https://github.com/mrdoob/three.js) | JavaScript | MIT | 133 | 133 | train | [LICENSES/mrdoob__three.js.txt](datasets/agent_commits/LICENSES/mrdoob__three.js.txt) |
| [mui/material-ui](https://github.com/mui/material-ui) | JavaScript | MIT | 72 | 72 | train | [LICENSES/mui__material-ui.txt](datasets/agent_commits/LICENSES/mui__material-ui.txt) |
| [pcottle/learnGitBranching](https://github.com/pcottle/learnGitBranching) | JavaScript | MIT | 38 | 38 | train | [LICENSES/pcottle__learnGitBranching.txt](datasets/agent_commits/LICENSES/pcottle__learnGitBranching.txt) |
| [phaserjs/phaser](https://github.com/phaserjs/phaser) | JavaScript | MIT | 20 | 20 | test | [LICENSES/phaserjs__phaser.txt](datasets/agent_commits/LICENSES/phaserjs__phaser.txt) |
| [playcanvas/engine](https://github.com/playcanvas/engine) | JavaScript | MIT | 133 | 133 | train | [LICENSES/playcanvas__engine.txt](datasets/agent_commits/LICENSES/playcanvas__engine.txt) |
| [preactjs/preact](https://github.com/preactjs/preact) | JavaScript | MIT | 39 | 39 | train | [LICENSES/preactjs__preact.txt](datasets/agent_commits/LICENSES/preactjs__preact.txt) |
| [quasarframework/quasar](https://github.com/quasarframework/quasar) | JavaScript | MIT | 35 | 35 | train | [LICENSES/quasarframework__quasar.txt](datasets/agent_commits/LICENSES/quasarframework__quasar.txt) |
| [react/react](https://github.com/react/react) | JavaScript | MIT | 61 | 61 | train | [LICENSES/react__react.txt](datasets/agent_commits/LICENSES/react__react.txt) |
| [sveltejs/kit](https://github.com/sveltejs/kit) | JavaScript | MIT | 88 | 88 | train | [LICENSES/sveltejs__kit.txt](datasets/agent_commits/LICENSES/sveltejs__kit.txt) |
| [sveltejs/svelte](https://github.com/sveltejs/svelte) | JavaScript | MIT | 40 | 40 | train | [LICENSES/sveltejs__svelte.txt](datasets/agent_commits/LICENSES/sveltejs__svelte.txt) |
| [vercel/next.js](https://github.com/vercel/next.js) | JavaScript | MIT | 116 | 116 | test | [LICENSES/vercel__next.js.txt](datasets/agent_commits/LICENSES/vercel__next.js.txt) |
| [wekan/wekan](https://github.com/wekan/wekan) | JavaScript | MIT | 133 | 133 | train | [LICENSES/wekan__wekan.txt](datasets/agent_commits/LICENSES/wekan__wekan.txt) |
| [3b1b/manim](https://github.com/3b1b/manim) | Python | MIT | 161 | 161 | test | [LICENSES/3b1b__manim.txt](datasets/agent_commits/LICENSES/3b1b__manim.txt) |
| [apache/airflow](https://github.com/apache/airflow) | Python | Apache-2.0 | 376 | 376 | train | [LICENSES/apache__airflow.txt](datasets/agent_commits/LICENSES/apache__airflow.txt) |
| [apache/superset](https://github.com/apache/superset) | Python | Apache-2.0 | 440 | 440 | train | [LICENSES/apache__superset.txt](datasets/agent_commits/LICENSES/apache__superset.txt) |
| [ArchiveBox/ArchiveBox](https://github.com/ArchiveBox/ArchiveBox) | Python | MIT | 255 | 255 | train | [LICENSES/ArchiveBox__ArchiveBox.txt](datasets/agent_commits/LICENSES/ArchiveBox__ArchiveBox.txt) |
| [commaai/openpilot](https://github.com/commaai/openpilot) | Python | MIT | 22 | 22 | test | [LICENSES/commaai__openpilot.txt](datasets/agent_commits/LICENSES/commaai__openpilot.txt) |
| [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | Python | Apache-2.0 | 237 | 237 | train | [LICENSES/deepset-ai__haystack.txt](datasets/agent_commits/LICENSES/deepset-ai__haystack.txt) |
| [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | Python | Apache-2.0 | 211 | 211 | train | [LICENSES/deepspeedai__DeepSpeed.txt](datasets/agent_commits/LICENSES/deepspeedai__DeepSpeed.txt) |
| [gradio-app/gradio](https://github.com/gradio-app/gradio) | Python | Apache-2.0 | 216 | 216 | train | [LICENSES/gradio-app__gradio.txt](datasets/agent_commits/LICENSES/gradio-app__gradio.txt) |
| [home-assistant/core](https://github.com/home-assistant/core) | Python | Apache-2.0 | 440 | 440 | train | [LICENSES/home-assistant__core.txt](datasets/agent_commits/LICENSES/home-assistant__core.txt) |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Python | Apache-2.0 | 440 | 440 | train | [LICENSES/huggingface__transformers.txt](datasets/agent_commits/LICENSES/huggingface__transformers.txt) |
| [huggingface/trl](https://github.com/huggingface/trl) | Python | Apache-2.0 | 11 | 11 | train | [LICENSES/huggingface__trl.txt](datasets/agent_commits/LICENSES/huggingface__trl.txt) |
| [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot) | Python | Apache-2.0 | 440 | 440 | train | [LICENSES/hummingbot__hummingbot.txt](datasets/agent_commits/LICENSES/hummingbot__hummingbot.txt) |
| [ipython/ipython](https://github.com/ipython/ipython) | Python | BSD-3-Clause | 239 | 239 | validation | [LICENSES/ipython__ipython.txt](datasets/agent_commits/LICENSES/ipython__ipython.txt) |
| [jax-ml/jax](https://github.com/jax-ml/jax) | Python | Apache-2.0 | 34 | 34 | train | [LICENSES/jax-ml__jax.txt](datasets/agent_commits/LICENSES/jax-ml__jax.txt) |
| [joke2k/faker](https://github.com/joke2k/faker) | Python | MIT | 9 | 9 | train | [LICENSES/joke2k__faker.txt](datasets/agent_commits/LICENSES/joke2k__faker.txt) |
| [keon/algorithms](https://github.com/keon/algorithms) | Python | MIT | 77 | 77 | train | [LICENSES/keon__algorithms.txt](datasets/agent_commits/LICENSES/keon__algorithms.txt) |
| [keras-team/keras](https://github.com/keras-team/keras) | Python | Apache-2.0 | 109 | 109 | train | [LICENSES/keras-team__keras.txt](datasets/agent_commits/LICENSES/keras-team__keras.txt) |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | Python | Apache-2.0 | 440 | 440 | validation | [LICENSES/mlflow__mlflow.txt](datasets/agent_commits/LICENSES/mlflow__mlflow.txt) |
| [netbox-community/netbox](https://github.com/netbox-community/netbox) | Python | Apache-2.0 | 103 | 103 | train | [LICENSES/netbox-community__netbox.txt](datasets/agent_commits/LICENSES/netbox-community__netbox.txt) |
| [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech) | Python | Apache-2.0 | 207 | 207 | train | [LICENSES/NVIDIA-NeMo__Speech.txt](datasets/agent_commits/LICENSES/NVIDIA-NeMo__Speech.txt) |
| [onnx/onnx](https://github.com/onnx/onnx) | Python | Apache-2.0 | 173 | 173 | train | [LICENSES/onnx__onnx.txt](datasets/agent_commits/LICENSES/onnx__onnx.txt) |
| [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | Python | BSD-3-Clause | 440 | 440 | train | [LICENSES/pandas-dev__pandas.txt](datasets/agent_commits/LICENSES/pandas-dev__pandas.txt) |
| [pypa/pipenv](https://github.com/pypa/pipenv) | Python | MIT | 239 | 239 | train | [LICENSES/pypa__pipenv.txt](datasets/agent_commits/LICENSES/pypa__pipenv.txt) |
| [ray-project/ray](https://github.com/ray-project/ray) | Python | Apache-2.0 | 440 | 440 | train | [LICENSES/ray-project__ray.txt](datasets/agent_commits/LICENSES/ray-project__ray.txt) |
| [saleor/saleor](https://github.com/saleor/saleor) | Python | BSD-3-Clause | 188 | 188 | train | [LICENSES/saleor__saleor.txt](datasets/agent_commits/LICENSES/saleor__saleor.txt) |
| [soxoj/maigret](https://github.com/soxoj/maigret) | Python | MIT | 19 | 19 | validation | [LICENSES/soxoj__maigret.txt](datasets/agent_commits/LICENSES/soxoj__maigret.txt) |
| [srbhr/Resume-Matcher](https://github.com/srbhr/Resume-Matcher) | Python | Apache-2.0 | 35 | 35 | validation | [LICENSES/srbhr__Resume-Matcher.txt](datasets/agent_commits/LICENSES/srbhr__Resume-Matcher.txt) |
| [streamlit/streamlit](https://github.com/streamlit/streamlit) | Python | Apache-2.0 | 440 | 440 | train | [LICENSES/streamlit__streamlit.txt](datasets/agent_commits/LICENSES/streamlit__streamlit.txt) |
| [tornadoweb/tornado](https://github.com/tornadoweb/tornado) | Python | Apache-2.0 | 81 | 81 | test | [LICENSES/tornadoweb__tornado.txt](datasets/agent_commits/LICENSES/tornadoweb__tornado.txt) |
| [walter201230/Python](https://github.com/walter201230/Python) | Python | CC-BY-4.0 | 1 | 1 | test | [LICENSES/walter201230__Python.txt](datasets/agent_commits/LICENSES/walter201230__Python.txt) |
| [zulip/zulip](https://github.com/zulip/zulip) | Python | Apache-2.0 | 362 | 362 | train | [LICENSES/zulip__zulip.txt](datasets/agent_commits/LICENSES/zulip__zulip.txt) |
| [ant-design/ant-design](https://github.com/ant-design/ant-design) | TypeScript | MIT | 452 | 452 | train | [LICENSES/ant-design__ant-design.txt](datasets/agent_commits/LICENSES/ant-design__ant-design.txt) |
| [ant-design/ant-design-pro](https://github.com/ant-design/ant-design-pro) | TypeScript | MIT | 132 | 132 | train | [LICENSES/ant-design__ant-design-pro.txt](datasets/agent_commits/LICENSES/ant-design__ant-design-pro.txt) |
| [apify/crawlee](https://github.com/apify/crawlee) | TypeScript | Apache-2.0 | 7 | 7 | train | [LICENSES/apify__crawlee.txt](datasets/agent_commits/LICENSES/apify__crawlee.txt) |
| [appsmithorg/appsmith](https://github.com/appsmithorg/appsmith) | TypeScript | Apache-2.0 | 135 | 135 | train | [LICENSES/appsmithorg__appsmith.txt](datasets/agent_commits/LICENSES/appsmithorg__appsmith.txt) |
| [BabylonJS/Babylon.js](https://github.com/BabylonJS/Babylon.js) | TypeScript | Apache-2.0 | 500 | 500 | validation | [LICENSES/BabylonJS__Babylon.js.txt](datasets/agent_commits/LICENSES/BabylonJS__Babylon.js.txt) |
| [backstage/backstage](https://github.com/backstage/backstage) | TypeScript | Apache-2.0 | 500 | 500 | train | [LICENSES/backstage__backstage.txt](datasets/agent_commits/LICENSES/backstage__backstage.txt) |
| [calcom/cal.diy](https://github.com/calcom/cal.diy) | TypeScript | MIT | 173 | 173 | train | [LICENSES/calcom__cal.diy.txt](datasets/agent_commits/LICENSES/calcom__cal.diy.txt) |
| [clauderic/dnd-kit](https://github.com/clauderic/dnd-kit) | TypeScript | MIT | 205 | 205 | train | [LICENSES/clauderic__dnd-kit.txt](datasets/agent_commits/LICENSES/clauderic__dnd-kit.txt) |
| [colinhacks/zod](https://github.com/colinhacks/zod) | TypeScript | MIT | 70 | 70 | train | [LICENSES/colinhacks__zod.txt](datasets/agent_commits/LICENSES/colinhacks__zod.txt) |
| [cypress-io/cypress](https://github.com/cypress-io/cypress) | TypeScript | MIT | 500 | 500 | train | [LICENSES/cypress-io__cypress.txt](datasets/agent_commits/LICENSES/cypress-io__cypress.txt) |
| [desktop/desktop](https://github.com/desktop/desktop) | TypeScript | MIT | 500 | 500 | train | [LICENSES/desktop__desktop.txt](datasets/agent_commits/LICENSES/desktop__desktop.txt) |
| [eggjs/egg](https://github.com/eggjs/egg) | TypeScript | MIT | 28 | 28 | train | [LICENSES/eggjs__egg.txt](datasets/agent_commits/LICENSES/eggjs__egg.txt) |
| [emberjs/ember.js](https://github.com/emberjs/ember.js) | TypeScript | MIT | 113 | 113 | test | [LICENSES/emberjs__ember.js.txt](datasets/agent_commits/LICENSES/emberjs__ember.js.txt) |
| [expo/expo](https://github.com/expo/expo) | TypeScript | MIT | 271 | 271 | test | [LICENSES/expo__expo.txt](datasets/agent_commits/LICENSES/expo__expo.txt) |
| [freeCodeCamp/freeCodeCamp](https://github.com/freeCodeCamp/freeCodeCamp) | TypeScript | BSD-3-Clause | 7 | 7 | train | [LICENSES/freeCodeCamp__freeCodeCamp.txt](datasets/agent_commits/LICENSES/freeCodeCamp__freeCodeCamp.txt) |
| [github/docs](https://github.com/github/docs) | TypeScript | CC-BY-4.0 | 80 | 80 | validation | [LICENSES/github__docs.txt](datasets/agent_commits/LICENSES/github__docs.txt) |
| [heroui-inc/heroui](https://github.com/heroui-inc/heroui) | TypeScript | Apache-2.0 | 136 | 136 | train | [LICENSES/heroui-inc__heroui.txt](datasets/agent_commits/LICENSES/heroui-inc__heroui.txt) |
| [homebridge/homebridge](https://github.com/homebridge/homebridge) | TypeScript | Apache-2.0 | 169 | 169 | train | [LICENSES/homebridge__homebridge.txt](datasets/agent_commits/LICENSES/homebridge__homebridge.txt) |
| [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio) | TypeScript | Apache-2.0 | 8 | 8 | train | [LICENSES/HumanSignal__label-studio.txt](datasets/agent_commits/LICENSES/HumanSignal__label-studio.txt) |
| [Kong/insomnia](https://github.com/Kong/insomnia) | TypeScript | Apache-2.0 | 88 | 88 | train | [LICENSES/Kong__insomnia.txt](datasets/agent_commits/LICENSES/Kong__insomnia.txt) |
| [mantinedev/mantine](https://github.com/mantinedev/mantine) | TypeScript | MIT | 103 | 103 | test | [LICENSES/mantinedev__mantine.txt](datasets/agent_commits/LICENSES/mantinedev__mantine.txt) |
| [microsoft/playwright](https://github.com/microsoft/playwright) | TypeScript | Apache-2.0 | 91 | 91 | test | [LICENSES/microsoft__playwright.txt](datasets/agent_commits/LICENSES/microsoft__playwright.txt) |
| [microsoft/vscode](https://github.com/microsoft/vscode) | TypeScript | MIT | 500 | 500 | train | [LICENSES/microsoft__vscode.txt](datasets/agent_commits/LICENSES/microsoft__vscode.txt) |
| [motiondivision/motion](https://github.com/motiondivision/motion) | TypeScript | MIT | 500 | 500 | train | [LICENSES/motiondivision__motion.txt](datasets/agent_commits/LICENSES/motiondivision__motion.txt) |
| [neoclide/coc.nvim](https://github.com/neoclide/coc.nvim) | TypeScript | MIT | 44 | 44 | train | [LICENSES/neoclide__coc.nvim.txt](datasets/agent_commits/LICENSES/neoclide__coc.nvim.txt) |
| [nestjs/nest](https://github.com/nestjs/nest) | TypeScript | MIT | 98 | 98 | train | [LICENSES/nestjs__nest.txt](datasets/agent_commits/LICENSES/nestjs__nest.txt) |
| [nrwl/nx](https://github.com/nrwl/nx) | TypeScript | MIT | 478 | 478 | train | [LICENSES/nrwl__nx.txt](datasets/agent_commits/LICENSES/nrwl__nx.txt) |
| [palantir/blueprint](https://github.com/palantir/blueprint) | TypeScript | Apache-2.0 | 56 | 56 | train | [LICENSES/palantir__blueprint.txt](datasets/agent_commits/LICENSES/palantir__blueprint.txt) |
| [payloadcms/payload](https://github.com/payloadcms/payload) | TypeScript | MIT | 226 | 226 | train | [LICENSES/payloadcms__payload.txt](datasets/agent_commits/LICENSES/payloadcms__payload.txt) |
| [pixijs/pixijs](https://github.com/pixijs/pixijs) | TypeScript | MIT | 160 | 160 | train | [LICENSES/pixijs__pixijs.txt](datasets/agent_commits/LICENSES/pixijs__pixijs.txt) |
| [portainer/portainer](https://github.com/portainer/portainer) | TypeScript | Zlib | 5 | 5 | train | [LICENSES/portainer__portainer.txt](datasets/agent_commits/LICENSES/portainer__portainer.txt) |
| [QwikDev/qwik](https://github.com/QwikDev/qwik) | TypeScript | MIT | 219 | 219 | test | [LICENSES/QwikDev__qwik.txt](datasets/agent_commits/LICENSES/QwikDev__qwik.txt) |
| [react-hook-form/react-hook-form](https://github.com/react-hook-form/react-hook-form) | TypeScript | MIT | 82 | 82 | test | [LICENSES/react-hook-form__react-hook-form.txt](datasets/agent_commits/LICENSES/react-hook-form__react-hook-form.txt) |
| [recharts/recharts](https://github.com/recharts/recharts) | TypeScript | MIT | 199 | 199 | train | [LICENSES/recharts__recharts.txt](datasets/agent_commits/LICENSES/recharts__recharts.txt) |
| [remix-run/react-router](https://github.com/remix-run/react-router) | TypeScript | MIT | 48 | 48 | test | [LICENSES/remix-run__react-router.txt](datasets/agent_commits/LICENSES/remix-run__react-router.txt) |
| [remix-run/remix](https://github.com/remix-run/remix) | TypeScript | MIT | 88 | 88 | train | [LICENSES/remix-run__remix.txt](datasets/agent_commits/LICENSES/remix-run__remix.txt) |
| [solidjs/solid](https://github.com/solidjs/solid) | TypeScript | MIT | 32 | 32 | train | [LICENSES/solidjs__solid.txt](datasets/agent_commits/LICENSES/solidjs__solid.txt) |
| [storybookjs/storybook](https://github.com/storybookjs/storybook) | TypeScript | MIT | 500 | 500 | train | [LICENSES/storybookjs__storybook.txt](datasets/agent_commits/LICENSES/storybookjs__storybook.txt) |
| [styled-components/styled-components](https://github.com/styled-components/styled-components) | TypeScript | MIT | 53 | 53 | train | [LICENSES/styled-components__styled-components.txt](datasets/agent_commits/LICENSES/styled-components__styled-components.txt) |
| [supabase/supabase](https://github.com/supabase/supabase) | TypeScript | Apache-2.0 | 500 | 500 | test | [LICENSES/supabase__supabase.txt](datasets/agent_commits/LICENSES/supabase__supabase.txt) |
| [super-productivity/super-productivity](https://github.com/super-productivity/super-productivity) | TypeScript | MIT | 500 | 500 | train | [LICENSES/super-productivity__super-productivity.txt](datasets/agent_commits/LICENSES/super-productivity__super-productivity.txt) |
| [transloadit/uppy](https://github.com/transloadit/uppy) | TypeScript | MIT | 44 | 44 | train | [LICENSES/transloadit__uppy.txt](datasets/agent_commits/LICENSES/transloadit__uppy.txt) |
| [trpc/trpc](https://github.com/trpc/trpc) | TypeScript | MIT | 58 | 58 | train | [LICENSES/trpc__trpc.txt](datasets/agent_commits/LICENSES/trpc__trpc.txt) |
| [ueberdosis/tiptap](https://github.com/ueberdosis/tiptap) | TypeScript | MIT | 214 | 214 | train | [LICENSES/ueberdosis__tiptap.txt](datasets/agent_commits/LICENSES/ueberdosis__tiptap.txt) |
| [vitejs/vite](https://github.com/vitejs/vite) | TypeScript | MIT | 81 | 81 | train | [LICENSES/vitejs__vite.txt](datasets/agent_commits/LICENSES/vitejs__vite.txt) |
| [vuejs/vitepress](https://github.com/vuejs/vitepress) | TypeScript | MIT | 173 | 173 | validation | [LICENSES/vuejs__vitepress.txt](datasets/agent_commits/LICENSES/vuejs__vitepress.txt) |
| [xtermjs/xterm.js](https://github.com/xtermjs/xterm.js) | TypeScript | MIT | 125 | 125 | train | [LICENSES/xtermjs__xterm.js.txt](datasets/agent_commits/LICENSES/xtermjs__xterm.js.txt) |

### Copyleft licenses

| Repository | Language | License | Human rows | AI rows | Split | License text |
|---|---|---|---:|---:|---|---|
| [2dust/v2rayN](https://github.com/2dust/v2rayN) | C# | GPL-3.0 | 16 | 16 | test | [copyleft/LICENSES/2dust__v2rayN.txt](datasets/agent_commits/copyleft/LICENSES/2dust__v2rayN.txt) |
| [jellyfin/jellyfin](https://github.com/jellyfin/jellyfin) | C# | GPL-2.0 | 16 | 16 | train | [copyleft/LICENSES/jellyfin__jellyfin.txt](datasets/agent_commits/copyleft/LICENSES/jellyfin__jellyfin.txt) |
| [QL-Win/QuickLook](https://github.com/QL-Win/QuickLook) | C# | GPL-3.0 | 6 | 6 | test | [copyleft/LICENSES/QL-Win__QuickLook.txt](datasets/agent_commits/copyleft/LICENSES/QL-Win__QuickLook.txt) |
| [espressif/arduino-esp32](https://github.com/espressif/arduino-esp32) | C++ | LGPL-2.1 | 88 | 88 | train | [copyleft/LICENSES/espressif__arduino-esp32.txt](datasets/agent_commits/copyleft/LICENSES/espressif__arduino-esp32.txt) |
| [FreeCAD/FreeCAD](https://github.com/FreeCAD/FreeCAD) | C++ | LGPL-2.1 | 43 | 43 | train | [copyleft/LICENSES/FreeCAD__FreeCAD.txt](datasets/agent_commits/copyleft/LICENSES/FreeCAD__FreeCAD.txt) |
| [NixOS/nix](https://github.com/NixOS/nix) | C++ | LGPL-2.1 | 110 | 110 | validation | [copyleft/LICENSES/NixOS__nix.txt](datasets/agent_commits/copyleft/LICENSES/NixOS__nix.txt) |
| [RPCS3/rpcs3](https://github.com/RPCS3/rpcs3) | C++ | GPL-2.0 | 9 | 9 | train | [copyleft/LICENSES/RPCS3__rpcs3.txt](datasets/agent_commits/copyleft/LICENSES/RPCS3__rpcs3.txt) |
| [telegramdesktop/tdesktop](https://github.com/telegramdesktop/tdesktop) | C++ | GPL-3.0 | 110 | 110 | train | [copyleft/LICENSES/telegramdesktop__tdesktop.txt](datasets/agent_commits/copyleft/LICENSES/telegramdesktop__tdesktop.txt) |
| [docmirror/dev-sidecar](https://github.com/docmirror/dev-sidecar) | JavaScript | MPL-2.0 | 16 | 16 | train | [copyleft/LICENSES/docmirror__dev-sidecar.txt](datasets/agent_commits/copyleft/LICENSES/docmirror__dev-sidecar.txt) |
| [jgraph/drawio-desktop](https://github.com/jgraph/drawio-desktop) | JavaScript | GPL-3.0 | 12 | 12 | train | [copyleft/LICENSES/jgraph__drawio-desktop.txt](datasets/agent_commits/copyleft/LICENSES/jgraph__drawio-desktop.txt) |
| [koodo-reader/koodo-reader](https://github.com/koodo-reader/koodo-reader) | JavaScript | AGPL-3.0 | 1 | 1 | train | [copyleft/LICENSES/koodo-reader__koodo-reader.txt](datasets/agent_commits/copyleft/LICENSES/koodo-reader__koodo-reader.txt) |
| [overleaf/overleaf](https://github.com/overleaf/overleaf) | JavaScript | AGPL-3.0 | 110 | 110 | train | [copyleft/LICENSES/overleaf__overleaf.txt](datasets/agent_commits/copyleft/LICENSES/overleaf__overleaf.txt) |
| [ToolJet/ToolJet](https://github.com/ToolJet/ToolJet) | JavaScript | AGPL-3.0 | 133 | 133 | train | [copyleft/LICENSES/ToolJet__ToolJet.txt](datasets/agent_commits/copyleft/LICENSES/ToolJet__ToolJet.txt) |
| [frappe/erpnext](https://github.com/frappe/erpnext) | Python | GPL-3.0 | 440 | 440 | train | [copyleft/LICENSES/frappe__erpnext.txt](datasets/agent_commits/copyleft/LICENSES/frappe__erpnext.txt) |
| [kovidgoyal/calibre](https://github.com/kovidgoyal/calibre) | Python | GPL-3.0 | 93 | 93 | validation | [copyleft/LICENSES/kovidgoyal__calibre.txt](datasets/agent_commits/copyleft/LICENSES/kovidgoyal__calibre.txt) |
| [kovidgoyal/kitty](https://github.com/kovidgoyal/kitty) | Python | GPL-3.0 | 128 | 128 | train | [copyleft/LICENSES/kovidgoyal__kitty.txt](datasets/agent_commits/copyleft/LICENSES/kovidgoyal__kitty.txt) |
| [paperless-ngx/paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) | Python | GPL-3.0 | 154 | 154 | validation | [copyleft/LICENSES/paperless-ngx__paperless-ngx.txt](datasets/agent_commits/copyleft/LICENSES/paperless-ngx__paperless-ngx.txt) |
| [Foundry376/Mailspring](https://github.com/Foundry376/Mailspring) | TypeScript | GPL-3.0 | 328 | 328 | train | [copyleft/LICENSES/Foundry376__Mailspring.txt](datasets/agent_commits/copyleft/LICENSES/Foundry376__Mailspring.txt) |
| [grafana/grafana](https://github.com/grafana/grafana) | TypeScript | AGPL-3.0 | 500 | 500 | train | [copyleft/LICENSES/grafana__grafana.txt](datasets/agent_commits/copyleft/LICENSES/grafana__grafana.txt) |
| [renovatebot/renovate](https://github.com/renovatebot/renovate) | TypeScript | AGPL-3.0 | 500 | 500 | train | [copyleft/LICENSES/renovatebot__renovate.txt](datasets/agent_commits/copyleft/LICENSES/renovatebot__renovate.txt) |
| [TriliumNext/Trilium](https://github.com/TriliumNext/Trilium) | TypeScript | AGPL-3.0 | 26 | 26 | train | [copyleft/LICENSES/TriliumNext__Trilium.txt](datasets/agent_commits/copyleft/LICENSES/TriliumNext__Trilium.txt) |

Repositories of the index whose license doesn't allow redistributing the rows (custom or source-available
terms, or no license) were skipped before any processing, so none of their code is in this dataset.

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
