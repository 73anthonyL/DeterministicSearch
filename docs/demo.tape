# Recorded with VHS (https://github.com/charmbracelet/vhs):
#   uv run --no-sync vhs docs/demo.tape      (writes docs/demo.gif)
# Uses a scratch library so the recording never touches ~/.dsearch.
Output docs/demo.gif

Set Shell bash
Set FontSize 13
Set Width 1200
Set Height 900
Set Framerate 10
Set PlaybackSpeed 1.0
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

Hide
Type "export DSEARCH_HOME=$(mktemp -d) && cd $(mktemp -d) && clear"
Enter
Sleep 500ms
Show

Type "dsearch demo"
Sleep 600ms
Enter
Sleep 14s
