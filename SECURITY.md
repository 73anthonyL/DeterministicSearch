# Security policy

## What dsearch does with your data

Everything runs on your machine. The only network access is the one-time
download of an embedding model from the Hugging Face Hub on first use; set
`HF_HUB_OFFLINE=1` to forbid even that. Your PDFs, the index built from them,
and your queries never leave the computer. The tool makes no API calls and
generates no text.

The index lives in `~/.dsearch` (or `$DSEARCH_HOME`) and contains the full
text of every source you add, in plain JSON. Treat that directory with the
same care as the PDFs themselves.

## Supported versions

Only the latest minor release receives fixes.

| Version | Supported |
|---|---|
| latest `0.x` | yes |
| older | no |

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting form:

**https://github.com/73anthonyL/DeterministicSearch/security/advisories/new**

You will get an acknowledgement within 7 days. Once a fix is ready it is
released as a patch version and credited to you in the changelog unless you
ask otherwise.

Things in scope: anything that lets a crafted PDF, EPUB, or other input run
code, read files outside the library, or write outside `DSEARCH_HOME`; and
anything that sends data off the machine. A PDF that merely extracts badly is
a bug, not a vulnerability; please open an ordinary issue for that.
