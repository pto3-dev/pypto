# Backend

Per-architecture behaviour, kept out of the passes.

Passes never branch on `BackendType`. Everything architecture-specific — codegen
target, runtime API names, hazard workarounds, cross-core layout rules — is answered
by a `BackendHandler` obtained from the active `PassContext`. Adding an architecture
means adding a handler, not editing passes.

| Page | What it covers |
| ---- | -------------- |
| [BackendHandler: principled backend dispatch](00-backend_handler.md) | The virtual interface, how passes query it, and what adding a new backend requires |

## See Also

- [Pass, PassContext, PassPipeline, and PassManager](../passes/00-pass_manager.md) — where the handler comes from.
- [PTO ISA reference](../../reference/index.md) — the hardware differences the handlers abstract over.

## Experimental Ascend310P target

`BackendType.Ascend310P` identifies the 310P/A1 target independently of
Ascend910B/A3. Its executable backend is not yet available. Selecting it through
`get_backend_instance`, `set_backend_type`, `ir.compile`, or `RunConfig(arch=...)` raises a
`ValueError` before code generation or runtime dispatch. Rejected global
selection preserves the previous configuration and allows a supported target to
be selected afterwards. Passing a legacy A3/A5 platform alongside an explicit 310P
target also raises instead of overriding it. Existing backend enum values remain unchanged.

This identifier is the first incremental step in A1 bring-up. It does not enable
PTOAS A1 code generation, ISA instructions, simulator execution, or the Simpler
runtime. Later features must supply those capabilities before removing the
selection guard; 310P must never silently fall back to the A3 backend.

`is_backend_available(BackendType.Ascend310P)` returns `False` without configuring
or instantiating a backend. Tools that offer selectable backends, including the
memory map viewer, use this query to omit experimental targets whose
implementation is unavailable. `BackendType.__members__` still lists all
recognized target identifiers.
