# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

<!--
Document your project's quality standards here.

Questions to answer:
- What patterns are forbidden?
- What linting rules do you enforce?
- What are your testing requirements?
- What code review standards apply?
-->

(To be filled by the team)

---

## Forbidden Patterns

<!-- Patterns that should never be used and why -->

(To be filled by the team)

---

## Required Patterns

<!-- Patterns that must always be used -->

### Preserve lightweight metadata before releasing large objects

When an experiment releases large raw DataFrame/list objects with `del` and
`gc.collect()`, extract any downstream metadata first, such as column names,
split sizes, or manifest fields.

**Why**: Several experiment scripts intentionally free raw MIMIC objects before
tokenization or evaluation to reduce memory pressure. Later code must not read
from those deleted variables.

**Wrong**
```python
del test_full_constants
constant_columns = test_full_constants[0].columns.tolist()
```

**Correct**
```python
if test_full_constants:
    constant_columns = test_full_constants[0].columns.tolist()
else:
    constant_columns = pd.read_csv(get_mimic_constants_path(), nrows=1).columns.tolist()
del test_full_constants
```

**Validation**: Run at least `python -m py_compile <changed_script.py>` or
`python -m compileall <changed_module_or_script>` after changing object
lifetime around long-running experiment scripts.

---

## Testing Requirements

<!-- What level of testing is expected -->

(To be filled by the team)

---

## Code Review Checklist

<!-- What reviewers should check -->

(To be filled by the team)
