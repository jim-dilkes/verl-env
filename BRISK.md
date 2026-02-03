<!-- brisk-session-manager -->
# Brisk Session Manager

## Session Start (AUTOMATIC)
On new conversation, agent automatically:
1. Read `.active-card` → get path to current kanban card
2. Read the card → understand feature scope, progress, decisions
3. Check `## Working Notes` → session handoff context
4. Summarize: "Resuming [feature]. Last: [X]. Next: [Y]"

If no `.active-card`: prompt user to run `/feat`
If auto-restore unclear: user can run `/continuesession` for explicit summary

## The Garden (.garden/)

Agent's persistent memory for this project. Lives in `.garden/` at repo root.

**Structure:**
```
.garden/
  docs-agent/       # Implementation knowledge (for agents)
  docs-user/        # Usage knowledge (for humans)
  kanban/           # Feature cards (MAIN BRANCH ONLY)
    0-intentions.md # User's high-level direction
    1-backlog/      # Planned work
    2-working/      # In progress
    3-completed/    # Done
```

**CRITICAL: Kanban lives in main branch only.**
- `.garden/kanban/` exists only in main repo, not in feature branches
- `.active-card` points to card path relative to main repo's kanban
- In worktrees: kanban accessed via `$MAIN_REPO/.garden/kanban/`
- Never copy/move kanban dirs into feature branches

### Garden Maintenance (EVERY COMMIT)

**Every commit MUST include docs changes.** The garden is worked gently but continuously.

**On each commit:**
- Update relevant docs-agent/ files with implementation knowledge gained
- Prune outdated information immediately—stale docs are worse than no docs
- Update card's Working Notes with progress/decisions
- Keep docs concise and informative; no filler

**Principles:**
- Nurture: Add knowledge deliberately, granular files, descriptive names
- Prune: Remove stale docs aggressively, archive completed cards periodically
- Use: Read liberally—exists to reduce code exploration
- Small, frequent updates > big infrequent rewrites

**File naming:** `<domain>/<subject>-<aspect>.md` (kebab-case, descriptive)

### STATUS.md Dashboard (WHEN WARRANTED)

**Location:** `~/Documents/PhD/Research Projects/<project>/{C} Kanban/STATUS.md`

**Agent identity:** Use "brisk" when updating last updated time/tracking.

**Garden principle: Nurture, don't clutter.**

**Update when closing features if work was substantive:**
- Major feature addition → yes
- Architecture change → yes
- Experiments launched → yes
- Minor fix/tweak → no

**What to update:**
- Status emoji/callout (🟢 active, 🟡 paused, etc.)
- Momentum bar (visual progress indicator)
- Status table (last activity date, current focus)
- Brief progress notes (1-2 lines, no filler)

**Integration:**
- Main dashboard: `~/Documents/PhD/Research Projects/STATUS.md` (embeds project sections)
- Update atomically with commit (not separate maintenance task)

## Core Behaviors
- Use AskUserQuestion tool for user input (never plain text questions)
- Commit frequently; remind user if uncommitted changes accumulate
- Keep card's Working Notes updated with timestamped entries
- Include docs changes in every commit

## Card Lifecycle
**During work:**
- Update `## Working Notes` after each significant change (timestamped)
- Check off `## Scope` items as completed
- Add discovered sub-tasks to scope
- Log key decisions: judgment calls, alternatives rejected

**Commits:**
- Commit frequently during work (no card move needed)
- Update Working Notes between commits
- Always include docs-agent/ or docs-user/ updates

**When all scope items complete:**
- Before final commit: "All done. Move card to completed?"
- If yes: move card, then commit

## Slash Commands
| Command | When to use |
|---------|------------|
| `/feat` | Start new feature - select card or create via interview |
| `/status` | Quick overview of card, git state, docs health |
| `/docsread` | Pull relevant docs into Working Notes for context |
| `/docsupdate` | Update .garden/docs-*/ after implementation |
| `/endsession` | Pause work - save state to Working Notes |
| `/close` | Complete feature - move card, update docs, merge |

## Project Structure
- `.active-card`: Pointer to kanban card in main repo (gitignored)
- `.garden/kanban/`: Cards in **main branch only** (1-backlog → 2-working → 3-completed)
- `.garden/docs-agent/`: Implementation knowledge
- `.garden/docs-user/`: Usage documentation

## Kanban Location (IMPORTANT)

**Default:** Kanban cards live in `.garden/kanban/` within the repo.

**External linking:** `.garden/kanban/` can be a symlink to an external location (e.g., Obsidian vault).
- Use `/link-kanban` to create symlink
- All operations resolve symlinks transparently
- `.active-card` stores absolute paths
- Worktrees copy the symlink, all share external location

**Path formats:**
- Display format: `.garden/kanban/2-working/card.md`
- Storage (.active-card): `/absolute/path/to/external/kanban/2-working/card.md`

**Location by context:**
- **In main repo:** `.garden/kanban/` (real directory or symlink to external)
- **In worktrees:** `.garden/kanban/` (symlink copied from main)
- **External (if linked):** e.g., `~/Documents/PhD/Research Projects/<project>/Development/{C} Kanban/`

External kanban separates code git and kanban git histories.
<!-- /brisk-session-manager -->
