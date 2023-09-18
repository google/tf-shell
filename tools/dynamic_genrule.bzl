"""
This module defines a rule which allows generating a dynamic set of output files
which is only known after running the generation command e.g. bazel's execution
stage. The output files can be used as targets for other rules.
"""

def _impl(ctx):
    tree = ctx.actions.declare_directory(ctx.attr.name)
    ctx.actions.run(
        inputs = [],
        outputs = [tree],
        arguments = [tree.path],
        progress_message = "Generating files into '%s'" % tree.path,
        executable = ctx.executable.tool,
    )

    return [DefaultInfo(files = depset([tree]))]

dynamic_genrule = rule(
    implementation = _impl,
    attrs = {
        "tool": attr.label(
            executable = True,
            cfg = "exec",
            allow_files = True,
        ),
    },
)
