# Phase 5 sources: licence review

Owner decision (2026-11-09): OWASP Cheat Sheets and MITRE CWE stay out of pretraining. They are reserved for Phase 5
security instruction tuning or retrieval, subject to this review. CyberNative DPO and CodeSearchNet stay excluded.
This is an engineering reading of the published terms, not legal advice.

| Source | Terms (checked 2026-11-10) | Commercial use | Conditions |
|---|---|---|---|
| OWASP Cheat Sheet Series | CC BY-SA 4.0 (`LICENSE.md` in `OWASP/CheatSheetSeries`) | Allowed | Credit, link to licence, mark changes; adapted material that is shared must stay CC BY-SA |
| MITRE CWE | CWE Terms of Use (cwe.mitre.org/about/termsofuse.html) | Allowed, royalty-free | Every copy keeps MITRE's copyright notice and the licence; CWE is a MITRE trademark |

## How each may be used

- **CWE: usable** for weakness classification, explanation and remediation rows, and for retrieval. Ship a
  `NOTICE` with MITRE's copyright designation and the terms in any dataset or index that contains CWE text. Use CWE
  IDs and names factually, with no implied endorsement.
- **OWASP: usable with share-alike.** Any instruction row that quotes or adapts cheat-sheet text is adapted material.
  If that dataset is ever published, it must be CC BY-SA 4.0 with attribution. Retrieval that shows excerpts must
  credit OWASP and link the licence. Keep OWASP-derived rows in their own file, tagged `CC-BY-SA-4.0`, so they can
  be dropped or relicensed separately from MIT/Apache rows.
- Neither source enters pretraining shards. Both get the same 13-gram decontamination against the frozen benchmarks
  and the behaviour suite as existing instruction data.
