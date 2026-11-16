# 同梱している公式のプロンプト整形資料

「LLMで整形」の「skillsを利用」で使う資料。各モデルの公式リポジトリから
内容を変えずに取得したもの（app/prompt_skills.py が読み込む）。いずれも
Apache License 2.0（`LICENSE-Apache-2.0.txt`）。

| フォルダ | 取得元 | コミット |
|---|---|---|
| `moss-image-prompt-enhancer/` | https://github.com/OpenMOSS/MOSS-Image `skills/moss-image-prompt-enhancer/`（SKILL.md、references/layout.md、scripts/compile_prompt.py） | 6be4453024c5acae9dcd0a191e26ee524e743310 |
| `krea-2/` | https://github.com/krea-ai/krea-2 `docs/expansion.txt` | db3984fbc6e13b34c0064990fc2d95ac64d00058 |
| `flux-3/` | https://github.com/black-forest-labs/flux-3-image-klein `src/flux3/pup_prompts.json`、`pup_prompts_external_vlm.json` | bfc8889a4497f846416f253a3972a1941ac698b9 |

MOSS-Image のものだけが Agent Skills 形式（SKILL.md）の「skill」。Krea-2 と
FLUX.3 には skill が無く、公式が配っているプロンプト拡張用のシステム
プロンプトを同じ仕組みで使う。

更新するときは同じパスのファイルを差し替え、上の表のコミットを書き換える。
