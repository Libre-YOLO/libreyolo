# Contributing

Thanks for your interest in contributing to LibreYOLO.

LibreYOLO is an MIT-licensed computer vision library focused on YOLO-style
training and inference. Contributions are welcome, but maintainer review time
is limited. Keep changes focused, tested, and aligned with the project.

## Licensing and provenance (non-negotiable)

LibreYOLO is MIT. That only stays true if every contribution is clean:

- Only submit code you wrote yourself, or code from a source explicitly
  licensed under MIT, Apache-2.0, BSD, or a similarly permissive license.
- Code derived from GPL, AGPL, LGPL, non-commercial, proprietary, or
  unknown-license sources is not accepted in any form. This includes
  rewrites, paraphrases, renamed variables, and "inspired by" adaptations:
  a derivative work stays a derivative work no matter how it is reworded.
- If your PR ports or adapts third-party code, declare it in the PR
  description: upstream repository, commit, and license. Ported code
  without a provenance declaration will not be merged.
- New ported code must update the notice files (`THIRD_PARTY_NOTICES.txt`
  and the per-family `NOTICE` convention under `libreyolo/models/`).
- By submitting a contribution you certify that you have the right to submit
  it under the MIT license, in the sense of the Developer Certificate of
  Origin (developercertificate.org).
- Licensing doubts are blocking, not nits. If you are unsure whether a
  source is compatible, ask in the issue before writing code.

## Before opening a PR

- For anything non-trivial, open an issue before submitting a PR so we can
  agree on the approach before review time is spent.
- PRs must link to an issue (for anything non-trivial)
- A good PR solves one clearly described problem.
- Use common sense and keep scope tight.
- Changes that add, remove, or reinterpret checkpoint metadata must update
  `docs/checkpoint_schema.md` and the shared helpers in
  `libreyolo/utils/serialization.py`.

## LLM policy

- Do not add LLMs as co-authors in commits.
- Do not open LLM-generated issues.
- Read AGENTS.md
