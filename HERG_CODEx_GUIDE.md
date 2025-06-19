##################  HERG / CODEx COLLABORATION GUIDE  ##################
This file is **instructions**, *not* code. Do **not** edit, remove, or
"implement" it. It tells any Codex-like agent HOW to act whenever it sees a
line that begins with

        ◇ CODEX_IMPLEMENT:

Context
-------
HERG is a self-evolving hyper-vector brain. The runtime—or a human—occasionally
appends new TODO markers so Codex can extend HERG’s capabilities. Your job as
Codex is to generate patches **only** for those TODO markers—nothing else.

Rules for every `◇ CODEX_IMPLEMENT:` marker
1. **Write executable Python 3.10 code** (or tests/docs) that fully meets the
   specification on that marker line and its surrounding comment.
2. **Leave this guide exactly as it is.** Do **not** treat this text as a
   marker to implement.
3. Return your work as a **unified diff** (`diff --git a/… b/…`) so the repo’s
   automation can apply it cleanly.
4. Use only the Python standard library plus the packages already imported in
   the target file. HERG currently allows:
      • numpy   (always)  
      • cupy    (optional GPU backend)  
      • openai  (only inside CI runners)  
      • gh CLI  (in automation scripts)
5. Ensure **all existing unit tests pass**; add tests for any new public
   function.
6. Respect HERG’s backend abstraction:  
      • CPU default via NumPy  
      • GPU via CuPy (or Torch if explicitly enabled)
#######################################################################
What this accomplishes

Puts the collaboration rules in-repo (Git-tracked).

README now points new contributors or LLMs straight to the guide.

Copilot / GPT models reading the repo will obey the guide and ignore it when generating patches.

Commit example:

bash
Copy
Edit
git add README.md HERG_CODEx_GUIDE.md
git commit -m "docs: add global Codex collaboration guide"
git push
