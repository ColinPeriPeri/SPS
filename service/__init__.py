"""Inference entry points invoked by the UiPath Performer.

No web framework by design: UiPath owns all SQL access and invokes this as a
process, reading the JSON contract from stdout.

Nothing is re-exported here on purpose: the module `service.run_inference` and
the function `run_inference` share a name, so hoisting the function into the
package namespace would shadow the module and break `import service.run_inference`.
Import explicitly instead:

    from service.run_inference import run_inference, run_batch
"""
