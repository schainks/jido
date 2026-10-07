# Which tool is right for this request?

The labelling rubric for `real_calls/`. A label says which kind of tool is the **right first action for the request**, judged from what
the request needs, not from which tool an assistant happened to use. Overlapping habits are merged: reading a file is `read_file` whether it
is done with a Read tool or `cat`; finding something is `search_files` whether it is `grep`, `find` or a glob.

Assume the assistant can do everything below. Pick the single most appropriate **first** action to begin the work.

| Label | Right when the request needs the assistant to... |
| --- | --- |
| `read_file` | look at the contents of a specific, known file or output (a file, a log, a config, a diff already named) |
| `search_files` | find where something is in a codebase or filesystem: a symbol, string, file, or which files match |
| `edit_file` | change an existing file |
| `write_file` | create a new file |
| `git` | do version control: status, diff, log, branch, commit, push, merge, rebase |
| `github` | work with GitHub: pull requests, issues, reviews, CI runs, releases |
| `run_code` | run tests, a build, a script, a package manager or a dev server, and see what happens |
| `shell_ops` | do a system or filesystem operation: create, copy, move or delete files, check processes, ports, disk, environment, or wait |
| `remote_or_http` | run something on another machine over ssh, or call an HTTP endpoint or API from the shell |
| `web_fetch` | retrieve the contents of a specific URL |
| `web_search` | search the web for information it does not have |
| `browser_or_app_ui` | operate a browser or an app's interface: navigate, click, type, take screenshots, drive a simulator |
| `delegate` | hand a self-contained or parallelisable piece of work to a subagent |
| `external_service` | use a connected service or integration: Slack, email, calendar, documents, analytics, error tracking, memory of past conversations |
| `use_skill` | invoke a named skill, slash command or plugin that the request itself asks for |
| `ask_user` | ask the user something first, because the request is too ambiguous or needs a decision before any action |
| `answer_directly` | no tool is needed: explain, summarise from what is already known, give an opinion, or reply |

## Rules

1. **Judge the request, not a guess at the assistant's habit.** "Show me the config" is `read_file` even if the assistant might `cat` it.
2. **First action, not the whole job.** "Fix the failing test" starts with `run_code` (see the failure) or `read_file`; choose the most direct first step and say how sure you are.
3. **Name the capability, not the loader.** If the request needs Slack, the label is `external_service`, even though a tool must be loaded first.
4. **A request that needs context you do not have** (a bare "yes", "continue", "do that") is not labelled; it was removed before you saw it. If one slips through, label `ask_user` with confidence `low`.
5. **When two labels are both defensible**, give the more direct one as `label` and the other as `alt`, and set `conf` to `med` or `low`.

Output one JSON object per request: `{"id": ..., "label": ..., "alt": null or a label, "conf": "high" | "med" | "low"}`.
