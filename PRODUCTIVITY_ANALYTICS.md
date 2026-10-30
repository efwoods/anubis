# Productivity Analytics — 2026

Measured 2026-10-27 on this machine (`/home/user`). Every figure below comes from a named local source; nothing is extrapolated.

## Summary

- **458,564 lines of source code** were added in Evan Woods's own commits in 2026 (2026-01-01 → 2026-10-27).
- **79.77%** of those lines (365,798) were committed on a day when Claude Code was in use. **20.23%** (92,766) have no Claude Code record.
- **30.95%** (141,941 lines) are in commits whose message carries a Claude `Co-Authored-By` trailer, meaning Claude made the commit.
- From the first Claude Code prompt on record (2026-05-14) onward, **93.06%** of committed code lines (365,798 of 393,060) fell on days with Claude Code activity.
- The only window with line-level measurement (2026-09-27 → 2026-10-27): Claude Code's file-writing tools wrote lines equal to **62.47%** of the code lines committed in the same repositories (10,543 of 16,878).
- A whole-year line-level split between Claude-written and Evan-written code is **not measured**. Claude Code keeps per-edit records only for the last 30 days, and git records who committed a line, not who typed the line.

## 1. Claude Code usage in 2026

Source: `~/.claude/history.jsonl` (one entry per typed prompt).

| Measure | Value |
|---|---|
| First prompt in 2026 | 2026-05-14 18:40:04 UTC |
| Last prompt at measurement time | 2026-10-27 18:44:22 UTC |
| Span | 166 days |
| Prompts typed in 2026 | 2,960 |
| Days with at least one prompt | 139 (UTC calendar days); 133 (America/New_York calendar days) |

Hours of activity, from session transcripts in `~/.claude/projects/` (2026-09-27 20:49 UTC → 2026-10-27, the only period with transcripts still on disk):

| Definition of "active" | Hours |
|---|---|
| Time between events separated by 30 minutes or less | 38.17 h |
| Time between events separated by 5 minutes or less | 21.16 h |

Hours for the full year are **not measured**: transcripts older than 30 days are deleted, and `history.jsonl` records prompt times, not session durations. `~/.claude/stats-cache.json` stopped updating on 2026-07-16 and covers only 2026-06-16 → 2026-07-16 (152 sessions, 20,744 messages).

## 2. Whole-year commit classification

### Method

- **Repositories:** every git repository under `/home/user` (depth 5) with Evan Woods's 2026 commits. Third-party clones with no Evan Woods commits (Cyberpunk 2077 mod forks, `obs-studio`, `agent-chat-ui`, vendored dependencies) contribute nothing.
- **Commits:** non-merge commits by `Evan Woods`, all branches, 2026-01-01 → 2026-10-27. Each commit hash is counted once, even when the same commit appears in a repository and in a `wt/f-*` clone of that repository.
- **Lines:** lines added (`git log --numstat`) in source-code files only: `.py .js .jsx .ts .tsx .c .h .cpp .hpp .cs .swift .kt .java .go .rs .sh .css .html .sql .reds .lua`. Excluded: markdown, lockfiles, notebooks, data dumps, `.min.js`, and any path under `node_modules/`, `site-packages/`, `.venv/`, `venv/`, `dist/`, `build/`, `vendor/`, `third_party/`. The exclusion removed one 614,500-line `node_modules` commit (`8ce60e3`, `anubis-discord`, 2026-07-15).
- **Claude Code activity per day:** a prompt in `history.jsonl` on the same America/New_York calendar day. A prompt is tied to a repository when the prompt's launch folder is inside the repository, or when the folder is `~/gh/anubis-project/wt/<name>` and exactly one repository owns a branch named `<name>`. 995 prompts map to a repository; 1,969 prompts were launched from parent folders (`~`, `~/gh/anubis-project/wt`, `~/gh/anubis-project`) and map to no repository.
- **Categories**, checked in this order:
  1. **Claude trailer** — the commit message matches `Co-Authored-By: Claude|Codex|Copilot|Cursor|GPT`, `noreply@anthropic.com`, or `Generated with Claude`.
  2. **Before first Claude Code record** — the commit predates 2026-05-14.
  3. **Same repository, same day** — Claude Code was used in the commit's repository that day.
  4. **Other folder, same day** — Claude Code was used that day, but from a folder that maps to no repository, or to a different repository.
  5. **No Claude Code prompt that day.**

### Results

| Category | Code lines | Share | Commits |
|---|---|---|---|
| Claude trailer | 141,941 | 30.95% | 122 |
| Same repository, same day | 126,191 | 27.52% | 156 |
| Other folder, same day | 97,666 | 21.30% | 101 |
| No Claude Code prompt that day (after 2026-05-14) | 27,262 | 5.95% | 15 |
| Before first Claude Code record (before 2026-05-14) | 65,504 | 14.28% | 466 |
| **Total** | **458,564** | **100.00%** | **860** |

Arithmetic:

- Claude Code activity that day = 141,941 + 126,191 + 97,666 = 365,798; 365,798 ÷ 458,564 = 79.77%
- No Claude Code record = 27,262 + 65,504 = 92,766; 92,766 ÷ 458,564 = 20.23%
- From 2026-05-14 onward = 458,564 − 65,504 = 393,060 lines; 365,798 ÷ 393,060 = 93.06%; 27,262 ÷ 393,060 = 6.94%

### By month

| Month | Code lines | Claude trailer | Same repository, same day | Other folder, same day | No prompt that day | Before first record | Share on Claude Code days |
|---|---|---|---|---|---|---|---|
| 2026-01 | 19,878 | 0 | 0 | 0 | 0 | 19,878 | no record |
| 2026-02 | 13,868 | 0 | 0 | 0 | 0 | 13,868 | no record |
| 2026-03 | 10,184 | 0 | 0 | 0 | 0 | 10,184 | no record |
| 2026-04 | 10,973 | 0 | 0 | 0 | 0 | 10,973 | no record |
| 2026-05 | 16,764 | 0 | 1,652 | 4,123 | 388 | 10,601 | 5,775 ÷ 16,764 = 34.45% |
| 2026-06 | 16,636 | 836 | 13,034 | 2,766 | 0 | 0 | 100.00% |
| 2026-07 | 50,115 | 239 | 33,280 | 16,596 | 0 | 0 | 100.00% |
| 2026-08 | 29,031 | 0 | 14,245 | 14,786 | 0 | 0 | 100.00% |
| 2026-09 | 281,921 | 139,621 | 62,120 | 53,306 | 26,874 | 0 | 255,047 ÷ 281,921 = 90.47% |
| 2026-10 | 9,194 | 1,245 | 1,860 | 6,089 | 0 | 0 | 100.00% |

September holds 281,921 ÷ 458,564 = 61.48% of the year's code lines.

### By repository

| Repository | Code lines | Claude trailer | Same repository, same day | Other folder, same day | No prompt that day | Before first record |
|---|---|---|---|---|---|---|
| `anubis` | 269,225 | 99,126 | 47,996 | 62,062 | 12,625 | 47,416 |
| `Neural-Nexus-Frontend` | 128,880 | 27,561 | 56,836 | 14,336 | 14,388 | 15,759 |
| `anubis-adapter` | 10,228 | 808 | 9,420 | 0 | 0 | 0 |
| `anubis-customer-portal` | 8,508 | 924 | 6,517 | 1,067 | 0 | 0 |
| `anubis-discord` | 8,174 | 3,480 | 0 | 4,694 | 0 | 0 |
| `minecraft-integration` | 7,139 | 0 | 1,227 | 5,912 | 0 | 0 |
| `anubis-mcp-server-ubuntu-desktop` | 7,112 | 3,983 | 211 | 2,918 | 0 | 0 |
| `anubis-slack` | 4,043 | 4,043 | 0 | 0 | 0 | 0 |
| `nn-streamlit-ui` (anubis-project) | 3,014 | 0 | 9 | 1,100 | 249 | 1,656 |
| `anubis-mcp-server-windows` | 2,978 | 0 | 2,978 | 0 | 0 | 0 |
| `~/.bin` | 2,021 | 0 | 0 | 1,767 | 0 | 254 |
| `anubis-twitch` | 1,933 | 0 | 220 | 1,713 | 0 | 0 |
| `prey-mods` | 1,624 | 0 | 0 | 1,624 | 0 | 0 |
| `anubis-mcp-server-mac-os` | 1,571 | 1,486 | 0 | 85 | 0 | 0 |
| `neural-nexus-custom-commits` | 843 | 0 | 777 | 0 | 0 | 66 |
| `anubis-mcp-server-mobile` | 530 | 530 | 0 | 0 | 0 | 0 |
| `anubis-scalable-server` | 388 | 0 | 0 | 388 | 0 | 0 |
| `db-migration` | 292 | 0 | 0 | 0 | 0 | 292 |
| `cloudflare_nn_local_tunnel` | 61 | 0 | 0 | 0 | 0 | 61 |
| **Total** | **458,564** | **141,941** | **126,191** | **97,666** | **27,262** | **65,504** |

### Largest commits

| Lines | Repository | Commit | Date | Category | Subject |
|---|---|---|---|---|---|
| 28,348 | `anubis` | `58c8219` | 2026-09-11 | Other folder, same day | major update |
| 12,486 | `anubis` | `ebbf502` | 2026-09-15 | No prompt that day | stable |
| 10,542 | `Neural-Nexus-Frontend` | `4f444c1` | 2026-09-11 | Same repository, same day | major update |
| 8,675 | `anubis` | `c8bd04c` | 2026-09-04 | Claude trailer | Stabilize the f-anubis worktree: ambient vision, summarization… |
| 8,654 | `Neural-Nexus-Frontend` | `fb17178` | 2026-09-04 | Same repository, same day | saving stable state |
| 8,306 | `Neural-Nexus-Frontend` | `a7d1855` | 2026-09-11 | Same repository, same day | major UI update |
| 7,599 | `Neural-Nexus-Frontend` | `11037e6` | 2026-09-06 | Claude trailer | Live voice mode, blocked-voice text fallback, emotion media… |
| 6,900 | `anubis` | `3c45e1c` | 2026-09-06 | Claude trailer | Voice hygiene, blocked-voice fallback, emotion media… |
| 6,697 | `Neural-Nexus-Frontend` | `98ec401` | 2026-09-15 | No prompt that day | aesthetic updates and navigation features |
| 6,436 | `Neural-Nexus-Frontend` | `cbafa53` | 2026-09-05 | Same repository, same day | stable working state |

## 3. Line-level measurement, last 30 days

Window: 2026-09-27 20:49 UTC (earliest transcript event on disk) → 2026-10-27.

- **Claude lines:** non-blank lines added by Claude Code `Write`, `Edit` and `MultiEdit` tool calls that did not return an error, read from every transcript in `~/.claude/projects/` including subagent transcripts. `Write` counts every non-blank line of the written content; `Edit` / `MultiEdit` count non-blank lines that `difflib.ndiff` marks as added between `old_string` and `new_string`.
- **Git lines:** lines added in Evan Woods's non-merge commits, all branches, same window, same source-code extensions.

| Repository | Lines Claude wrote | Lines added in commits | Claude share |
|---|---|---|---|
| `anubis` | 7,085 | 11,907 | 7,085 ÷ 11,907 = 59.50% |
| `prey-mods` | 1,572 | 1,624 | 1,572 ÷ 1,624 = 96.80% |
| `Neural-Nexus-Frontend` | 1,163 | 2,382 | 1,163 ÷ 2,382 = 48.82% |
| `minecraft-integration` | 702 | 906 | 702 ÷ 906 = 77.48% |
| `~/.bin` | 21 | 59 | 21 ÷ 59 = 35.59% |
| **Total** | **10,543** | **16,878** | **10,543 ÷ 16,878 = 62.47%** |

A further 748 Claude-written lines went to repositories with no Evan Woods commits in the window (`~`, `CyberpunkNewGamePlus`, `let_there_be_flight`, `mod_settings`, `red4ext.sdk`, `SharedPunk`, `cp2077-codeware`) and are left out of the total.

## 4. Limits of the measurement

- **Committed on a Claude Code day does not mean Claude-written.** A commit on a day with Claude Code activity can contain lines Evan typed. In the one window with line-level data, Claude's tool calls account for 62.47% of committed lines, not 100%.
- **No Claude Code record does not mean Evan-written.** The data covers Claude Code on this machine only. ChatGPT, Copilot, Codex, claude.ai, or Claude Code on another computer leave no record here; that applies to all of 2026-01 through 2026-04.
- **The 2026-09-15 commits are unexplained.** `history.jsonl` holds no prompts from 2026-09-13 through 2026-09-16, yet `ebbf502` (12,486 lines) and `98ec401` (6,697 lines) were committed on 2026-09-15. Whether those commits hold earlier, uncommitted Claude work is not verified.
- **The 62.47% figure can be off in both directions.** Not counted: code Claude wrote through shell commands (heredocs, `sed`), and code from other tools. Over-counted: Claude edits later reverted, rewritten or never committed, and lines Claude rewrote more than once.
- **Commit trailers undercount Claude.** Most Claude-written code was committed under Evan Woods's name without a trailer. `anubis` has 1,502 commits and 75 commit-message lines mentioning Claude.

## 5. Making the split measurable from now on

Keep the `Co-Authored-By: Claude …` trailer on every commit that contains Claude-written code. With that trailer in place, `git blame` attributes each surviving line to a Claude-assisted or a non-Claude commit, for any period, without depending on Claude Code's 30-day transcript retention.

## Reproducing

The two scripts were run from the session scratchpad, which is deleted when the session ends:

- `year_analysis_deduplicated.py` — sections 2 (whole-year classification)
- `llm_share_code_only.py` — section 3 (line-level, last 30 days)

Both read only `~/.claude/history.jsonl`, `~/.claude/projects/**/*.jsonl` and `git log --numstat` output, and write nothing.
