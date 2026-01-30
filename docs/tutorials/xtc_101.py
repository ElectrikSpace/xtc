import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _(mo):
    # === Utility functions for the tutorial ===
    from io import StringIO
    from contextlib import redirect_stderr, redirect_stdout
    import traceback
    from typing import Any
    import queue
    import time
    import multiprocessing as mp
    import exec_utils

    def get_backend_import(backend_name: str) -> str:
        """Return the import statement for the selected backend."""
        if backend_name == "MLIR":
            return "from xtc.backends.mlir import Backend"
        else:
            return "from xtc.backends.tvm import Backend"

    def get_print_opts_str(output_option: str) -> str:
        """Return the compiler print options string for the selected output."""
        opts_map = {
            "Source IR": "print_source_ir=True",
            "Transformed IR": "print_transformed_ir=True",
            "Lowered IR": "print_lowered_ir=True",
            "Assembly": "print_assembly=True",
        }
        return opts_map.get(output_option, "")

    def get_print_opts_dict(output_option: str) -> dict:
        """Return the compiler print options as a dictionary."""
        opts_map = {
            "Source IR": {"print_source_ir": True},
            "Transformed IR": {"print_transformed_ir": True},
            "Lowered IR": {"print_lowered_ir": True},
            "Assembly": {"print_assembly": True},
        }
        return opts_map.get(output_option, {})

    def get_backend_class(backend_name: str):
        """Return the Backend class for the selected backend."""
        if backend_name == "MLIR":
            from xtc.backends.mlir import Backend
        else:
            from xtc.backends.tvm import Backend
        return Backend

    def get_output_options(backend_name: str) -> list:
        """Return available output options based on backend (TVM doesn't support Lowered IR)."""
        if backend_name == "TVM":
            return ["Source IR", "Transformed IR", "Assembly"]
        else:
            return ["Source IR", "Transformed IR", "Lowered IR", "Assembly"]

    def create_backend_radio(label: str = "Backend:"):
        """Create a radio button for backend selection."""
        return mo.ui.radio(options=["MLIR", "TVM"], value="MLIR", label=label)

    def create_output_radio(backend_name: str, label: str = "Output options:"):
        """Create a radio button for output options based on backend."""
        return mo.ui.radio(
            options=get_output_options(backend_name),
            value="Assembly",
            label=label
        )

    def execute_editor_code(editor_value: str, display_results_fn=None, initial_namespace=None):
        """
        Execute editor code with stdout/stderr capture.
        Returns (success, output, captured_data).
        - If display_results_fn is provided, it's injected as 'display_results' in the namespace.
        - If initial_namespace is provided, those values are injected before execution.
        - captured_data contains any data captured via display_results.
        """
        captured = {"perf": 0.0}

        def _display_results(perf):
            captured["perf"] = perf

        namespace = dict(initial_namespace) if initial_namespace else {}
        if display_results_fn is not None:
            namespace["display_results"] = _display_results

        code_stderr = StringIO()
        code_stdout = StringIO()

        try:
            with redirect_stderr(code_stderr), redirect_stdout(code_stdout):
                exec(editor_value, namespace)
            output = code_stderr.getvalue() + code_stdout.getvalue()
            return True, output, captured
        except Exception:
            return False, traceback.format_exc(), captured

    def render_editor_output(success: bool, output: str, captured: dict):
        """Render the output of an editor execution as marimo elements."""
        if not success:
            return mo.md(f"**Code error:**\n```\n{output}\n```")

        perf_display = mo.md(f"**Performance:** {captured['perf']:.2f}% of peak")
        code_content = mo.md(f"```asm\n{output}\n```") if output else mo.md("*No IR output.*")
        code_accordion = mo.accordion({"Generated Code": code_content})
        return mo.vstack([perf_display, code_accordion])

    def run_exploration(generator, get_info=None):
        """
        Run exploration with progress bar. Returns sorted results.
        Generator should yield (index, total, sample_or_name, perf) tuples.
        """
        results = []
        best_sample = None
        best_perf = 0.0
        captured_info = {}

        first_result = next(generator, None)
        if first_result is None:
            return [], {}

        idx, total, sample, perf = first_result
        results.append({"sample": sample, "perf": perf})
        if perf > best_perf:
            best_perf = perf
            best_sample = sample

        with mo.status.progress_bar(total=total, title="Exploring schedules...", remove_on_exit=False) as progress:
            progress.update(
                title=f"Exploring schedules... (best: {best_perf:.1f}%)",
                subtitle=f"Sample {idx + 1}/{total}: {sample} -> {perf:.1f}%"
            )

            for idx, total, sample, perf in generator:
                results.append({"sample": sample, "perf": perf})

                if perf > best_perf:
                    best_perf = perf
                    best_sample = sample

                progress.update(
                    title=f"Exploring schedules... (best: {best_perf:.1f}%)",
                    subtitle=f"Sample {idx + 1}/{total}: {sample} -> {perf:.1f}%"
                )

        if get_info:
            captured_info.update(get_info())

        sorted_results = sorted(results, key=lambda x: x["perf"], reverse=True)
        return sorted_results, captured_info

    def start_streaming_execution(
        *,
        code: str,
        out: Any,
        throttle_s: float = 0.5,
    ) -> Any:
        """
        Start a marimo Thread that launches a subprocess and streams its output.
        Cancellation: if the spawning cell is invalidated, thread.should_exit becomes True,
        and we terminate the subprocess.
        """
        def target():
            import marimo as mo
            thread = mo.current_thread()

            ctx = mp.get_context("spawn")
            out_q: "mp.Queue" = ctx.Queue()
            p = ctx.Process(
                target=exec_utils._child_exec,
                args=(code, out_q),
                daemon=True
            )
            p.start()

            buf: list[str] = []
            last = 0.0

            def render(force: bool = False):
                nonlocal last
                now = time.time()
                if force or (now - last) >= throttle_s:
                    out.replace(
                        mo.md(
                            f"**Output:**\n\n```text\n{''.join(buf)}\n```"
                        )
                    )
                    last = now

            try:
                out.replace(mo.md("**Output:**\n\n```text\n\n```"))
                render(force=True)

                while True:
                    # Cancel requested? (cell invalidated by Cancel click / rerun / interrupt)
                    if thread.should_exit:
                        buf.append("\n[Cancelled]\n")
                        render(force=True)
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=1)
                        break

                    # Process ended?
                    if not p.is_alive():
                        # Drain remaining queue chunks
                        while True:
                            try:
                                kind, payload = out_q.get_nowait()
                            except queue.Empty:
                                break
                            if kind == "chunk":
                                buf.append(payload)
                        render(force=True)
                        break

                    # Get output chunk (non-blocking-ish)
                    try:
                        kind, payload = out_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if kind == "chunk":
                        buf.append(payload)
                        render()
                    elif kind == "done":
                        # allow loop to observe process exit / drain remaining data
                        continue
            finally:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)

        t = mo.Thread(target=target, daemon=True)
        t.start()
        return t

    return (
        get_backend_import,
        get_print_opts_str,
        get_print_opts_dict,
        get_backend_class,
        get_output_options,
        create_backend_radio,
        create_output_radio,
        execute_editor_code,
        render_editor_output,
        run_exploration,
        traceback,
        start_streaming_execution,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XTC Tutorial

    Welcome to the XTC tutorial! This interactive notebook will guide you through the fundamentals of performance engineering using XTC, a research platform for optimizing AI operators.

    By the end of this notebook, you will understand how to:
    - Define computational graphs with XTC
    - Compile and evaluate operator performance
    - Explore the scheduling space to find optimal configurations
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Installation

    Before starting, ensure you have XTC properly installed on your system.
    Please [follow the README](https://github.com/xtc-tools/xtc/blob/main/README.md).
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Define Your First Graph with XTC

    In XTC, computations are represented as **dataflow graphs**. A graph consists of:
    - **Nodes** representing operations each implementing a specific computation (Operators)
    - **Edges** representing data dependencies between operations (through Tensors)
    
    Where:
    - **Tensors** are multi-dimensional arrays that hold data
    - **Operators** are tensor operations (e.g., matrix multiplication, convolution)

    Let us start by creating a simple matrix multiplication graph. Matrix multiplication (matmul) computes $C = A \times B$ where:
    - $A$ is an $I \times K$ matrix
    - $B$ is a $K \times J$ matrix
    - $C$ is the resulting $I \times J$ matrix
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The code below is editable and the output (i.e. the serialized graph) is dynamically computed.

    1. Try modifying the dimensions or data type (float32, float64) to see how the graph changes!
    2. Add a ReLU activation after the matrix multiplication. Hint: `O.matmul()` returns a tensor that can be passed to another operator. Using ReLu: `R = O.relu(inp, name)`.
    """)
    return

@app.cell
def _(mo):
    def_editor = mo.ui.code_editor(
        value=
'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         C = O.matmul(a, b, name="C")
   return gb.graph

I, J, K, dtype = 4, 32, 512, "float32"
graph = matmul_graph(I=I,J=J,K=K,dtype=dtype)

print(graph)''',
        language="python",
        label=""
    )
    def_editor
    return def_editor,

@app.cell
def _(def_editor, execute_editor_code, mo):
    _success, _output, _ = execute_editor_code(def_editor.value)
    mo.md(f"**{'Output' if _success else 'Error'}:**\n```\n{_output}\n```")

## Section 3 - Compile and Evaluate

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compile and Evaluate

    Now that we have a graph, we can compile it and measure its baseline performance (without any optimization).

    The compilation pipeline in XTC follows these steps:
    1. **Create a Backend**: In XTC, the backend corresponds to an existing framework such as MLIR or TVM that, given a schedule, can generate the code for a specific target
    2. **Get a Scheduler**: In XTC, a scheduler is a builder that creates a schedule. Even without optimizations, we need a scheduler to get a default loop structure.
    3. **Compile**: Generate executable code
    4. **Evaluate**: Run the compiled code and measure performance
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The script below compiles the matmul graph without any optimization and displays both the generated code (if you unroll the accordion) and its performance on chip.

    1. Use the radio buttons to select the backend (MLIR or TVM) and which IR to display.
    2. *Inspect the generated code.* Look at the Source IR, Transformed IR, Lowered IR, and Assembly. Are you able to identify the different loops i, j, k ?
    3. *Observe the performance.* In your opinion, why is it so poor?

    **Note:** Performance is measured as a percentage of peak (the theoretical FLOP/s of the CPU). The latter is by default approximated by running a benchmark, but this technique comes with (little) noise. If you already know the peak perf of your machine in GFlops/s, you can set the variable `peak_flops` yourself.
    """)
    return

@app.cell
def _(mo):
    compile_editor = mo.ui.code_editor(
        value='''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
import xtc.runtimes.host.runtime as rt

# Problem setup
I, J, K, dtype = 4, 32, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I, J, K, dtype)
backend = Backend(graph)

# Compile (no transformations, just default loop structure)
scheduler = backend.get_scheduler()
scheduler.set_dims(['i','j','k'])
schedule = scheduler.schedule()

compiler = backend.get_compiler(dump_file="matmul", shared_lib=True, **print_opts)
module = compiler.compile(schedule)

# Evaluate and display results
peak_flops = rt.evaluate_flops(dtype)
evaluator = module.get_evaluator()
results, _, _ = evaluator.evaluate()
perf = (I * J * K) / min(results) / peak_flops * 100

display_results(perf)
''',
        language="python",
        label=""
    )
    compile_editor
    return compile_editor,

@app.cell
def _(create_backend_radio):
    compile_backend_radio = create_backend_radio()
    return compile_backend_radio,

@app.cell
def _(compile_backend_radio, create_output_radio, mo):
    compile_output_radio = create_output_radio(compile_backend_radio.value)
    mo.hstack([compile_backend_radio, compile_output_radio], justify="start", gap=4)
    return compile_output_radio,

@app.cell
def _(compile_backend_radio, compile_editor, compile_output_radio, execute_editor_code, get_backend_class, get_print_opts_dict, mo, render_editor_output):
    _namespace = {
        "Backend": get_backend_class(compile_backend_radio.value),
        "print_opts": get_print_opts_dict(compile_output_radio.value),
    }
    _success, _output, _captured = execute_editor_code(compile_editor.value, display_results_fn=True, initial_namespace=_namespace)
    mo.stop(not _success, mo.md(f"**Code error:**\n```\n{_output}\n```"))
    render_editor_output(_success, _output, _captured)
    return


## Section 4 - Optimize with Scheduling

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Optimize with Scheduling

    The baseline compilation produces correct but unoptimized code. To improve performance, XTC exposes a **scheduler** with imperative primitives that transform the loop nest:

    - `sch.interchange(["i", "k", "j"])` reorders the loops. Interchange improves memory access patterns by ensuring stride-1 access (contiguous memory) rather than strided access, maximizing cache efficiency. Along with primitive `sch.strip_mine` (see below) this allows to actually tile the loop body.
    - `sch.strip_mine("j", {"j1": 16})` breaks loop `j` into smaller chunks of size 16. This transformation can also be seen as 1D tiling.
    - To perform actual multi-dimensional tiling, combine `sch.strip_mine` with `sch.interchange`. For example, 2D tiling:
      ```python
      sch.strip_mine("j", {"j1": 16})
      sch.strip_mine("k", {"k1": 16})
      sch.interchange(["i", "j", "k", "j1", "k1"])
      ```
      This creates $16 \times 16$ tiles over the $(j, k)$ dimensions.
    - `sch.vectorize(["j1"])` vectorizes the computation along the loop `j1`. Vectorization uses SIMD instructions to process multiple elements in parallel, significantly increasing throughput on modern CPUs.
    - `sch.unroll({"j1":1})` unrolls the loop `j1` with an unroll factor of 1 (which has no effect). Unrolling reduces loop overhead (fewer branches) and exposes more instruction-level parallelism to the hardware.
    - `sch.parallelize(["i"])` execute in parallel along the loop `i`. Parallelization is only useful on a multiple cores architecture and actually splits and dispatch the work onto the available cores.
    - `sch.split("j", {"j0": 0, "j1": 16})` splits `j` into two segments, creating `j0` (iterations 0-15) and `j1` (iterations 16+). Splitting is useful for applying different transformations to different parts of a loop (e.g., vectorize the main part, handle the remainder separately).
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The code below lets you define a schedule using imperative primitives. Use the radio buttons to select the backend and what IR to display. Compare the performance and generated code with the unoptimized version from the previous section.

    1. *Transform the code.* Start with simple transformations and build up.
    2. *Inspect the generated code.* How does the Assembly or IR differ from the baseline?
    3. *Try to maximize the performance!*

    *Hint: You may want to create a register tile (i1,j1), j1 being a multiple of your SIMD registers size.*
    """)
    return

@app.cell
def _(mo):
    sched_editor = mo.ui.code_editor(
        value='''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
import xtc.runtimes.host.runtime as rt

# Problem setup
I, J, K, dtype = 4, 32, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I, J, K, dtype)
backend = Backend(graph)

# Schedule definition
def schedule(sch):
   """Apply transformations to the scheduler."""
   sch.set_dims(['i','j','k'])
   # Add transformations here, e.g.:
   # sch.strip_mine("j", {"j1": 4})
   # sch.vectorize(["j1"])
   # sch.interchange(["i", "k", "j", "j1"])

# Compile
scheduler = backend.get_scheduler()
schedule(scheduler)
sched = scheduler.schedule()

compiler = backend.get_compiler(dump_file="matmul", shared_lib=True, **print_opts)
module = compiler.compile(sched)

# Evaluate and display results
peak_flops = rt.evaluate_flops(dtype)
evaluator = module.get_evaluator()
results, _, _ = evaluator.evaluate()
perf = (I * J * K) / min(results) / peak_flops * 100

display_results(perf)
''',
        language="python",
        label=""
    )
    sched_editor
    return sched_editor,

@app.cell
def _(create_backend_radio):
    sched_backend_radio = create_backend_radio()
    return sched_backend_radio,

@app.cell
def _(sched_backend_radio, create_output_radio, mo):
    sched_output_radio = create_output_radio(sched_backend_radio.value)
    mo.hstack([sched_backend_radio, sched_output_radio], justify="start", gap=4)
    return sched_output_radio,

@app.cell
def _(sched_backend_radio, sched_editor, sched_output_radio, execute_editor_code, get_backend_class, get_print_opts_dict, mo, render_editor_output):
    _namespace = {
        "Backend": get_backend_class(sched_backend_radio.value),
        "print_opts": get_print_opts_dict(sched_output_radio.value),
    }
    _success, _output, _captured = execute_editor_code(sched_editor.value, display_results_fn=True, initial_namespace=_namespace)
    mo.stop(not _success, mo.md(f"**Code error:**\n```\n{_output}\n```"))
    render_editor_output(_success, _output, _captured)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Define a schedule declaratively

    XTC allows you to describe the target loop structure using a Python dictionary. Instead of manually specifying each transformation step, you declare the desired final loop structure, and XTC automatically infers the sequence of transformations needed to achieve it.

    For example, the following loop structure:
    ```
    for i in ...
      for k in ...
        for j in ...
          for j1 in range(16):  // vectorized
    ```

    Can be described as:
    ```python
    {
      "i": {},
        "k": {},
          "j": {},
            "j#16": {"vectorize": True}
    }
    ```

    The dictionary keys define the loop order (outer to inner), `j#16` creates a tile of size 16 on `j`, and the `{"vectorize": True}` attribute marks that inner loop for vectorization.

    The declarative API supports several key optimizations:

    | Transformation      | Syntax                          |
    |---------------------|---------------------------------|
    | **Tiling**          | `"axis#size"`                   |
    | **Vectorization**   | `{"vectorize": True}`           |
    | **Parallelization** | `{"parallelize": True}`         |
    | **Unrolling**       | `{"unroll": factor}`            |
    | **Interchange**     | Key ordering in the dictionnary |
    | **Splitting**       | `"axis[beg:end]"`               |
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The code below lets you define a schedule specification and see the generated assembly. Try replicating the good-enough schedule you discovered in the previous section!
    """)
    return

@app.cell
def _(mo):
    descript_editor = mo.ui.code_editor(
        value='''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
from xtc.schedules.descript import descript_scheduler
import xtc.runtimes.host.runtime as rt

# Problem setup
I, J, K, dtype = 4, 32, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I, J, K, dtype)
backend = Backend(graph)

# Schedule specification
schedule_spec = {
   "i": {},
   "j": {},
   "k": {}
}

# Compile
scheduler = backend.get_scheduler()
descript_scheduler(
   scheduler=scheduler,
   node_name="C",
   abstract_axis=["i", "j", "k"],
   spec=schedule_spec
)
schedule = scheduler.schedule()

compiler = backend.get_compiler(dump_file="matmul", shared_lib=True, **print_opts)
module = compiler.compile(schedule)

# Evaluate and display results
peak_flops = rt.evaluate_flops(dtype)
evaluator = module.get_evaluator()
results, _, _ = evaluator.evaluate()
perf = (I * J * K) / min(results) / peak_flops * 100

display_results(perf)
''',
        language="python",
        label=""
    )
    descript_editor
    return descript_editor,

@app.cell
def _(create_backend_radio):
    descript_backend_radio = create_backend_radio()
    return descript_backend_radio,

@app.cell
def _(descript_backend_radio, create_output_radio, mo):
    descript_output_radio = create_output_radio(descript_backend_radio.value)
    mo.hstack([descript_backend_radio, descript_output_radio], justify="start", gap=4)
    return descript_output_radio,

@app.cell
def _(descript_backend_radio, descript_editor, descript_output_radio, execute_editor_code, get_backend_class, get_print_opts_dict, mo, render_editor_output):
    _namespace = {
        "Backend": get_backend_class(descript_backend_radio.value),
        "print_opts": get_print_opts_dict(descript_output_radio.value),
    }
    _success, _output, _captured = execute_editor_code(descript_editor.value, display_results_fn=True, initial_namespace=_namespace)
    mo.stop(not _success, mo.md(f"**Code error:**\n```\n{_output}\n```"))
    render_editor_output(_success, _output, _captured)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Experimenting with Multiple Schedules

    Performance engineering is often about exploring different optimization strategies. Different schedules can have dramatically different performance depending on:
    - **Problem size**: Small matrices may not benefit from parallelization overhead
    - **Hardware**: Cache sizes, vector width, and core count affect optimal tiling
    - **Data layout**: Memory access patterns influence cache efficiency

    In this section, we'll write a simple loop to try several schedule configurations and compare their performance.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The code below defines several schedule configurations using our declarative scheduler. You can:

    - **Add/modify configurations**: Try different tile sizes, loop orderings, or optimization combinations. The pre-built search space (variable `configurations` in function `explore` and lines above) may be poorly designed...
    - **Change the acquisition function**: Add caching, error handling, or custom metrics
    - **Modify the exploration loop**: Add early stopping or custom filtering

    The `explore()` function must `yield` tuples of `(index, total, config_name, performance)` for real-time progress display.
    """)
    return

@app.cell
def _(mo, run_exploration):
    explore_schedules_code = mo.ui.code_editor(
        value=
'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
from xtc.backends.tvm import Backend as TVM_Backend
from xtc.backends.mlir import Backend as MLIR_Backend
from xtc.schedules.descript import descript_scheduler
import xtc.runtimes.host.runtime as rt
from io import StringIO
from contextlib import redirect_stderr
import itertools

# Problem setup
I, J, K, dtype = 4, 32, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I=I, J=J, K=K, dtype=dtype)
peak_flops = rt.evaluate_flops(dtype)

# Evaluation helpers
def apply_schedule(graph, backend_cls, spec):
   """Apply a declarative schedule specification and compile."""
   backend = backend_cls(graph)
   scheduler = backend.get_scheduler()
   descript_scheduler(
         scheduler=scheduler,
         node_name="C",
         abstract_axis=["i", "j", "k"],
         spec=spec
   )
   schedule = scheduler.schedule()
   comp = backend.get_compiler(dump_file="test_mlir", shared_lib=True)
   code = StringIO()
   with redirect_stderr(code):
         module = comp.compile(schedule)
   return module, code.getvalue()

def evaluate(module, peak_flops, nfmadds):
   """Evaluate module performance as percentage of peak."""
   evaluator = module.get_evaluator()
   results, _, _ = evaluator.evaluate()
   result = min(results)
   time_flops = nfmadds / result
   perf = time_flops / peak_flops * 100
   return perf

def acquire(i1: int, j1: int, backend):
   """Evaluate a single configuration and return its performance."""
   module, _ = apply_schedule(
         graph=graph,
         backend_cls=backend,
         spec={
               "i": {},
               "j": {},
               "k": {},
               f"i#{i1}": {"unroll": True},
               f"j#{j1}": {"vectorize": True}
         },
   )
   return evaluate(module, peak_flops, I*J*K)

# Exploration loop
def explore():
   """Generator that yields (index, total, name, perf) for each evaluation."""
   i1_range = range(1, 5)
   j1_range = range(1, 5)
   backends = {'mlir': MLIR_Backend}
   configurations = list(itertools.product(i1_range, j1_range, backends.items()))
   total = len(configurations)

   for idx, (i1, j1, (name,backend)) in enumerate(configurations):
         perf = acquire(i1=i1, j1=j1, backend=backend)
         yield (idx, total, f"{name}:{i1}x{j1}", perf)

# Run exploration (displays a progress bar and returns sorted results)
def get_info():
   return {"dims": f"{I}x{J}x{K}", "dtype": dtype}

results = run_exploration(explore(), get_info)
''',
        language="python",
        label=""
    )
    run_explore_button = mo.ui.run_button(label="Run exploration")
    mo.vstack([explore_schedules_code, run_explore_button])
    return explore_schedules_code, run_explore_button

@app.cell
def _(explore_schedules_code, run_explore_button, mo, run_exploration, traceback):
    mo.stop(not run_explore_button.value, mo.md("*Click 'Run exploration' to execute the code.*"))

    # Execute the user's code with run_exploration available
    _namespace = {"run_exploration": run_exploration}
    try:
        exec(explore_schedules_code.value, _namespace)
    except Exception:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{traceback.format_exc()}\n```"))

    _results, _captured_info = _namespace.get("results", ([], {}))

    # Build results summary
    if not _results:
        mo.stop(True, mo.md("**Error:** No results returned. Make sure to call `results = run_exploration(explore(), get_info)`"))

    _dims = _captured_info.get("dims", "?")
    _dtype = _captured_info.get("dtype", "?")
    _baseline_perf = _results[-1]["perf"] if _results else 1.0
    _best = _results[0]

    _summary_lines = [
        f"### Schedule Exploration Results",
        f"",
        f"- **Problem:** {_dims} matmul, {_dtype}",
        f"",
        f"**Total configurations evaluated:** {len(_results)}",
        f"",
        f"#### Top 10 configurations:",
        f"",
        f"| Rank | Configuration | Performance | vs Baseline |",
        f"|------|---------------|-------------|-------------|",
    ]

    for _rank, _r in enumerate(_results[:10], 1):
        _speedup = _r["perf"] / _baseline_perf if _baseline_perf > 0 else 0
        _summary_lines.append(f"| {_rank} | {_r['sample']} | {_r['perf']:.2f}% | {_speedup:.2f}x |")

    _summary_lines.extend([
        f"",
        f"#### Best configuration: **{_best['sample']}** ({_best['perf']:.2f}% of peak)",
    ])

    mo.md("\n".join(_summary_lines))

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Automated Schedule Search with Strategies

    In the previous section, you manually defined schedule configurations to explore. While this approach works for small experiments, the scheduling space for real-world operators is vast—thousands or even millions of valid configurations exist. Manually exploring this space is impractical.

    **XTC provides scheduling strategies** that automate the exploration of the scheduling space. Strategies:

    1. **Define a structured search space**: Instead of arbitrary combinations, strategies encode domain knowledge about effective tilings, loop orders, and optimizations.
    2. **Filter invalid configurations**: Strategies automatically prune configurations that violate common sense constraints (e.g., unroll factors too large).
    3. **Support exhaustive and random sampling**: You can enumerate all valid schedules or sample randomly for faster exploration.
    4. **Encode best practices**: Each strategy implements proven optimization patterns from the literature (e.g., Goto-style, Ansor-style).

    **Why Use Strategies?**

    - *Efficiency*: A strategy like `Strategy_OO` reduces a matmul's search space from millions of arbitrary combinations to ~100-1000 valid, hardware-aware configurations.
    - *Reproducibility*: Strategies provide a systematic way to explore and compare optimizations.
    - *Automation*: Strategies integrate with XTC's search infrastructure to find optimal schedules automatically.
    
    **Available strategies:**

    | Strategy Name                       | Description                                                       | Use Case                          |
    |-------------------------------------|-------------------------------------------------------------------|-----------------------------------|
    | `Strategy_OO`                       | One-level tiling in O order (outer P, R, inner P)                 | Simple exploration, good baseline |
    | `Strategy_PRP`                      | P-R-P tiling (parallels, reductions, parallels)                   | Cache-friendly matmul             |
    | `Strategy_P1` / `Strategy_P1_v`     | One-level tiling with permutation (v = vectorization constrained) | Exploring loop orderings          |
    | `Strategy_PPRPRP`                   | Ansor-style multi-level tiling                                    | Advanced auto-tuning              |
    | `Strategy_PPRPRPv`                  | Ansor-style with vectorization constraint                         | Ensuring SIMD usage               |
    | `Strategy_PPRPRP_vr`                | Ansor-style with register and cache constraints                   | Hardware-aware tuning             |
    | `Strategy_PPWRPRP`                  | Ansor-style with write buffer                                     | Reducing memory traffic           |
    | `Strategy_GOTO` / `Strategy_GOTO_R` | Goto-style BLAS tiling (r = reduced search space)                 | High-performance GEMM             |

    **Naming convention**:
    - `P` = Parallel axes (e.g., i, j in matmul)
    - `R` = Reduction axes (e.g., k in matmul)
    - `O` = Outer parallel first, then reductions, then remaining parallels
    - `W` = Write buffer insertion point
    - `_v` suffix = Vectorization constraint (inner axis must be vectorizable)
    - `_vr` suffix = Vectorization + register/cache constraints
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The code below demonstrates using `Strategy_OO` to automatically explore the scheduling space. The code defines an `explore()` generator function that you can fully customize:

    - **Change the strategy**: Try `Strategy_PRP`, `Strategy_PPRPRP`, or `Strategy_GOTO`
    - **Modify the search loop**: Add early stopping, custom filtering, etc.
    - **Customize the acquisition function**: The `acquire()` function evaluates a sample - you can add caching, error handling, or custom metrics

    The `explore()` function must `yield` tuples of `(index, total, sample, performance)` for real-time progress display.
    """)
    return

@app.cell
def _(mo, run_exploration):
    strategy_editor = mo.ui.code_editor(
        value=
'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
from xtc.backends.mlir import Backend
from xtc.search.strategies import Strategy_OO
import xtc.runtimes.host.runtime as rt
from io import StringIO
from contextlib import redirect_stderr

# Problem setup
I, J, K, dtype = 64, 128, 256, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I=I, J=J, K=K, dtype=dtype)
peak_flops = rt.evaluate_flops(dtype)

# Strategy selection (try: Strategy_PRP, Strategy_PPRPRP, Strategy_GOTO, etc.)
max_unroll, vec_size = 64, 16
strategy = Strategy_OO(graph, max_unroll=max_unroll, vec_size=vec_size)

# Evaluation helpers
def compile_schedule(backend, sched):
   """Compile a schedule and return the module."""
   comp = backend.get_compiler(dump_file="test_mlir", shared_lib=True)
   code = StringIO()
   with redirect_stderr(code):
         module = comp.compile(sched)
   return module, code.getvalue()

def evaluate(module, peak_flops, nfmadds):
   """Evaluate module performance as percentage of peak."""
   evaluator = module.get_evaluator()
   results, _, _ = evaluator.evaluate()
   result = min(results)
   time_flops = nfmadds / result
   perf = time_flops / peak_flops * 100
   return perf

def acquire(sample):
   """Evaluate a single sample and return its performance."""
   impl = Backend(graph)
   scheduler = impl.get_scheduler()
   strategy.generate(scheduler, sample)
   schedule = scheduler.schedule()
   module, _ = compile_schedule(backend=impl, sched=schedule)
   return evaluate(module, peak_flops, I*J*K)

# Exploration loop
def explore():
   """Generator that yields (index, total, sample, perf) for each evaluation."""
   # Choose: strategy.exhaustive() or strategy.sample(num=N, seed=S)
   # samples = list(strategy.exhaustive())
   samples = list(strategy.sample(num=30, seed=42))
   total = len(samples)

   for idx, sample in enumerate(samples):
         perf = acquire(sample)
         yield (idx, total, sample, perf)

# Run exploration (displays a progress bar and returns sorted results)
def get_info():
   return {
         "dims": f"{I}x{J}x{K}",
         "dtype": dtype,
         "strategy": strategy.__class__.__name__,
         "strategy_params": f"max_unroll={max_unroll}, vec_size={vec_size}",
         "stats": dict(strategy.stats),
   }

results = run_exploration(explore(), get_info)
''',
        language="python",
        label=""
    )
    run_strategy_button = mo.ui.run_button(label="Run strategy exploration")
    mo.vstack([strategy_editor, run_strategy_button])
    return strategy_editor, run_strategy_button

@app.cell
def _(strategy_editor, run_strategy_button, mo, run_exploration, traceback):
    mo.stop(not run_strategy_button.value, mo.md("*Click 'Run strategy exploration' to execute the code.*"))

    # Execute the user's code with run_exploration available
    _namespace = {"run_exploration": run_exploration}
    try:
        exec(strategy_editor.value, _namespace)
    except Exception:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{traceback.format_exc()}\n```"))

    _results, _captured_info = _namespace.get("results", ([], {}))

    # Build results summary
    if not _results:
        mo.stop(True, mo.md("**Error:** No results returned. Make sure to call `results = run_exploration(explore(), get_info)`"))

    _dims = _captured_info.get("dims", "?")
    _dtype = _captured_info.get("dtype", "?")
    _strategy_name = _captured_info.get("strategy", "?")
    _strategy_params = _captured_info.get("strategy_params", "")
    _stats = _captured_info.get("stats", {})
    _best = _results[0]

    _summary_lines = [
        f"### Strategy Exploration Results",
        f"",
        f"- **Problem name:** {_dims} matmul, {_dtype}",
        f"- **Strategy:** {_strategy_name} ({_strategy_params})",
        f"- **Search space stats:** {_stats}",
        f"",
        f"**Total configurations evaluated:** {len(_results)}",
        f"",
        f"#### Top 10 configurations:",
        f"",
        f"| Rank | Sample | Performance |",
        f"|------|--------|-------------|",
    ]

    for _rank, _r in enumerate(_results[:10], 1):
        _summary_lines.append(f"| {_rank} | `{_r['sample']}` | {_r['perf']:.2f}% |")

    _summary_lines.extend([
        f"",
        f"#### Best schedule found:",
        f"",
        f"- **Sample:** `{_best['sample']}`",
        f"- **Performance:** {_best['perf']:.2f}% of peak",
    ])

    mo.md("\n".join(_summary_lines))

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Optimizing for larger matmul size and compare with Numpy

    In previous sections we explored small matmul problem sizes where tiling for
    multiple cache levels is not really necessary.

    Let's now try to optimize for instance a $1024 \times 1024 \times 1024$ matmul which
    should put more challenge on optimizing the memory cache reuse.
    Indeed, the size of each matrice is then 4MB for float32 which is generally much
    larger than the L2 cache on a CPU (typically 1MB).

    First, in order to have a comparison with an existing implementation,
    let's try the Python Numpy package matmul which is implemented through the OpenBLAS library.

    Note that OpenBLAS will use the number of available cores on the system, hence we must adjust
    the estimation of the `peak_flops` by setting the `cores` variable to the real number of
    cores (set to 4 here by default), change this for your architecture.
    """)
    return

@app.cell
def _(mo):
    numpy_editor = mo.ui.code_editor(
        value=
'''# Run numpy matmul and collect elapsed time
import timeit
import numpy as np
import xtc.runtimes.host.runtime as rt

I, J, K, dtype = 1024, 1024, 1024, "float32"

cores = 4    # set to an estimated number of core
peak_flops = cores * rt.evaluate_flops(dtype, threads=cores)

A = np.random.rand(I, K).astype(dtype)
B = np.random.rand(K, J).astype(dtype)
C = np.empty((I, J), dtype=dtype)

number = 5
elapsed = timeit.timeit(lambda: np.matmul(A, B, C), number=number)/number
perf = (I * J * K) / elapsed / peak_flops * 100
print(f"numpy perf: {perf}%")''',
        language="python",
        label=""
    )
    numpy_editor
    return numpy_editor,

@app.cell
def _(numpy_editor, execute_editor_code, mo):
    _success, _output, _ = execute_editor_code(numpy_editor.value)
    mo.stop(not _success, mo.md(f"**Code error:**\n```\n{_output}\n```"))
    mo.md(f"**Output:**\n```\n{_output}\n```")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Generally on a 4 cores architecture, the computed `numpy perf` should be
    around 50-70% of the peak parallel performance.

    You may use the sandbox below or develop on your own editor to generate an
    XTC schedule which is at least as good as this result.

    Here are some hints:
    - in order to reach performance given the parallel peak flops computed above,
    one will have to use the `parallelize` primitive or pass to strategies the additional
    argument `threads=cores` (to activate parallelization of the outer axes);
    - for large matmul sizes a good strategy to start with is `Strategy_GOTO`, in this case prefer the TVM backend (currently XTC does not implement packing primitives for MLIR);
    - the base strategy is generally not sufficient, and one may filter the generated
    samples to reduce the search space and converge faster, we've seen that the inner tile
    dimension, vectorization and unrolling are important, this can be done by eviction.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sandbox

    This place is your playground. It is where you will achieve the greatest challenges,
    using XTC from scratch ;)

    For instance, you could try to solve the challenge in section 8 above and reach the
     OpenBLAS performance on a large matmul.

    """)
    return

@app.cell
def _(mo):
    sandbox_editor = mo.ui.code_editor(
        value=
'''\
# Implement your challenge here!
print("Hello XTC!")''',
        language="python",
        label=""
    )
    run = mo.ui.run_button(label="Run sandbox")
    cancel = mo.ui.button(label="Stop execution", kind="danger")

    mo.vstack(
        [
            sandbox_editor,
            mo.hstack([run, cancel]),
        ]
    )
    return sandbox_editor, run, cancel

@app.cell
def _(mo, sandbox_editor, run, cancel, start_streaming_execution):
    # This output placeholder is what the background thread updates.
    out = mo.output
    out

    # If cancel is clicked, this cell reruns; that invalidates the previous run-cell,
    # making the old background thread's `should_exit` flip to True, and it will terminate.
    if cancel.value:
        out.replace(mo.md("Cancelled (if something was running, it will stop)."))
    else:
        mo.stop(not run.value, mo.md("*Click 'Run sandbox' to execute the code, and 'Stop execution' to cancel long runs.*"))
        # Start background execution. It will keep streaming until done or cancelled.
        start_streaming_execution(code=sandbox_editor.value, out=out, throttle_s=0.05)
    return

if __name__ == "__main__":
    app.run()
