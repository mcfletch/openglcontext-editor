# Clean-Room Procedure for Reading Copyleft Source

Every project in this workspace ships under BSD-style terms. Copyleft-licensed
code — GPL, LGPL, AGPL, SSPL, CC-BY-SA and anything else whose terms propagate —
therefore cannot be copied into them, in whole, in part, or in translation.

This document is binding on **all** work that would involve reading a copyleft
codebase, whether done by a person or by an agent.

## Rule 0: prefer a source that is not copyleft

**Reading copyleft source is the last resort, not the first move.** Before
invoking the procedure below, look for the same facts in a source whose licence
poses no problem. In rough order of preference:

1. **A published specification or format document** — a standards document, a
   vendor spec, a format reference, an RFC.
2. **The bytes themselves.** File formats can be read out of a hex dump and a
   handful of sample files. Interpretation derived from data you possess is
   yours.
3. **Observed behaviour.** Run the original program, measure what it does, write
   down the numbers. Black-box observation is clean by construction.
4. **A permissively licensed implementation or wiki** — a BSD/MIT/Apache
   reimplementation, or community format documentation (check the wiki's own
   licence; many are CC-BY-SA, which is copyleft for the *text* but does not
   taint facts you restate in your own words).
5. **Copyleft source** — only when the fact exists nowhere else, and only under
   the wall below.

Record which of these you used. "I went straight to the GPL source" is not an
acceptable answer when a format document existed.

## The wall

Two roles, and **nobody holds both**:

| Role | May read copyleft source | May write project code |
|---|---|---|
| **Reader** | yes | **never** |
| **Implementer** | **never** | yes |

Exactly one artifact crosses between them: **a specification file**, written by
the Reader in the Reader's own words. The Implementer reads the spec and
nothing else. No transcripts, no quotes, no "here's roughly what it does"
in conversation, no screenshots.

When the Implementer is an agent, the Reader must be a **separate sub-agent**
whose only output is the spec file. The sub-agent's own final report crosses the
wall too, so it must contain nothing but a path and a summary — see the brief
below.

## The Reader's brief

A Reader sub-agent must be given all of the following. Copy it into the prompt.

> You are acting as the **Reader** in a clean-room procedure. You may read the
> copyleft source at `<path>`. You will produce **one specification file** at
> `<spec path>`, and that file is the only thing anyone downstream will ever see.
>
> **Your spec MUST contain only facts required for compatibility:**
> - named constants and their numeric values (bit flags, magic numbers, defaults)
> - binary layouts: field order, types, sizes, endianness, offsets
> - identifiers that are part of the *interface vocabulary* (format field names,
>   entity keys, classnames) — the words a third party must use to interoperate
> - numeric relationships stated as mathematics (e.g. "velocity = direction ×
>   speed × 10"), in ordinary notation, not as code
> - externally observable behaviour, stated as behaviour
>
> **Your spec MUST NOT contain:**
> - source code, in any language, in any quantity — not even one line
> - pseudocode, or prose that walks through the implementation's control flow
>   statement by statement
> - the source's comments, docstrings, or documentation text, quoted or
>   paraphrased closely
> - internal function, variable, struct or file names, except where citing a file
>   as provenance
> - the source's structure: its function decomposition, its ordering of checks,
>   its choice of helper routines, its error-handling shape
> - anything creative rather than factual: tuned constant tables, hand-authored
>   data, heuristics that are a design choice rather than a format requirement
>
> Write every fact in your own words, in the smallest form that makes it usable.
> If you cannot state a fact without reproducing expression, **do not state it**
> — list it under "Excluded" and say why, so a human can decide.
>
> **Your final report must contain no source code and no new facts** — only the
> path you wrote, the number of facts recorded, and anything you excluded or
> flagged. Everything of substance goes in the spec file.

## Spec format

Each spec is a Markdown file in `specs/`, named `SPEC-<area>.md`. It opens with
a provenance header so the paper trail is self-contained:

```markdown
# SPEC-<area>: <what this describes>

| | |
|---|---|
| Source consulted | <project name and URL> |
| Licence of source | <e.g. GPLv2> |
| Version / commit | <exact commit hash and date> |
| Files consulted | <paths within that tree> |
| Non-copyleft sources checked first | <what you tried, and why it was insufficient> |
| Reader | <agent or person> |
| Date | <ISO date> |

## Scope
<what a consumer of this spec is trying to build>

## Facts
<numbered, each independently checkable>

## Excluded
<anything deliberately left out, and why>
```

Facts are numbered so code and review can cite them precisely (`SPEC-BSP38 §4.2`).

## The Implementer's rules

- Read the spec. Do not open the copyleft tree, and do not ask anyone what is in
  it.
- Cite the **spec**, not the original source, in comments and docstrings.
- If the spec is missing a fact you need, say so and request a spec revision.
  Do not go and look.
- If you have *already* read the source before the wall was raised, you are
  tainted for that area — see below.

## Escalation

Stop and ask a human when:

- a needed fact cannot be separated from its expression;
- the material looks creative rather than factual (a tuned table, an artistic
  asset, a heuristic that is a design decision);
- the licence is unclear, or the tree mixes licences;
- the "fact" is really an algorithm with no other reasonable implementation, and
  reproducing it would reproduce the original's expression.

## Retrofitting: when source was already read

A wall built after the fact is weaker than one built first, and the record should
say so plainly rather than imply otherwise. When an area was implemented by
someone who had read the source:

1. A **fresh Reader** who has not seen the implementation produces the spec from
   the source, per the brief above.
2. The existing implementation is compared against the spec. Every constant,
   layout and behaviour in the code must be traceable to a numbered fact.
3. Anything in the code that the spec does *not* justify is treated as suspect:
   remove it, or re-derive it, or escalate.
4. The spec header records that this was a retrofit, and which files were already
   written when it ran.

This gives an auditable provenance record and catches expression that leaked in.
It does not turn a retrofit into a true clean-room derivation, and the header
must not claim otherwise.

## Checklist

Before merging work that touched a copyleft codebase:

- [ ] A non-copyleft source was looked for first, and the result recorded.
- [ ] The Reader and the Implementer were different agents or people.
- [ ] A spec exists in `specs/` with a complete provenance header.
- [ ] The spec contains no code, no pseudocode, no source comments, no internal
      names, no structural description.
- [ ] Every derived constant and layout in the code cites a spec fact.
- [ ] Code comments cite the spec, not the copyleft file.
- [ ] Escalations, if any, were resolved by a human.
