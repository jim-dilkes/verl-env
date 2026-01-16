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

## Core Behaviors
- Use AskUserQuestion tool for user input (never plain text questions)
- Commit frequently; remind user if uncommitted changes accumulate
- Keep card's Working Notes updated with timestamped entries

## Card Lifecycle
**During work:**
- Update `## Working Notes` after each significant change (timestamped)
- Check off `## Scope` items as completed
- Add discovered sub-tasks to scope
- Log key decisions: judgment calls, alternatives rejected

**Commits:**
- Commit frequently during work (no card move needed)
- Update Working Notes between commits

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
- `.active-card`: Pointer to current kanban card (gitignored)
- `.garden/kanban/`: Cards (1-backlog → 2-working → 3-completed)
- `.garden/docs-agent/`: Implementation knowledge
- `.garden/docs-user/`: Usage documentation
<!-- /brisk-session-manager -->
