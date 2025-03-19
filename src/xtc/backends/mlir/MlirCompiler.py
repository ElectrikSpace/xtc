#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any, cast, TypeAlias
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path


from mlir.passmanager import PassManager
from mlir.dialects import transform
from mlir.dialects.transform import (
    NamedSequenceOp,
    structured,
    vector,
    get_parent_op,
)
from mlir.dialects.transform.structured import structured_match
from mlir.dialects.transform.loop import loop_unroll
from mlir.ir import (
    Location,
    InsertionPoint,
    UnitAttr,
    OpResult,
)


from xtc.utils.xdsl_aux import brand_inputs_with_noalias

from xtc.utils.tools import (
    get_mlir_prefix,
)

from xtc.utils.ext_tools import (
    transform_opts,
    lowering_opts,
    mlirtranslate_opts,
    llc_opts,
    opt_opts,
    shared_lib_opts,
    exe_opts,
    runtime_libs,
    system_libs,
    dump_file,
    mlirrunner_opts,
    objdump_bin,
    objdump_arm_bin,
    cc_bin,
    objdump_opts,
    objdump_color_opts,
)

from xtc.targets.host import HostModule

import xtc.backends.mlir as backend
import xtc.itf as itf

from .MlirModule import MlirModule, RawMlirModule
from .MlirScheduler import MlirSchedule
from .MlirNodeScheduler import MlirNodeSchedule


class MlirCompiler(itf.comp.Compiler):
    def __init__(
        self,
        implementer: "backend.MlirImplementer",
        **kwargs: Any,
    ):
        self._implementer = implementer
        self._compiler_kwargs = kwargs

    @property
    @override
    def implementer(self) -> itf.impl.Implementer:
        return self._implementer

    @override
    def compile(
        self,
        schedule: itf.schd.Schedule,
    ) -> itf.comp.Module:
        shared_lib = self._compiler_kwargs.get("shared_lib", False)
        executable = self._compiler_kwargs.get("executable", False)
        dump_file = self._compiler_kwargs.get("dump_file")
        temp_dir = None
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/{self._implementer.payload_name}"
            self._compiler_kwargs["dump_file"] = dump_file
        module = self.generate_module()
        compiler = MlirModuleCompiler(
            mlir_module=module,
            mlir_schedule=cast(MlirSchedule, schedule),
            concluding_passes=self._implementer.concluding_passes,
            always_vectorize=self._implementer.always_vectorize,
            **self._compiler_kwargs,
        )
        assert compiler.dump_file is not None
        compiler.compile()
        executable = HostModule(
            Path(compiler.dump_file).name,
            self._implementer.payload_name,
            f"{compiler.dump_file}.so",
            "shlib",
            bare_ptr=compiler.bare_ptr,
            np_inputs_spec=self._implementer.np_inputs_spec,
            np_outputs_spec=self._implementer.np_outputs_spec,
            reference_impl=self._implementer.reference_impl,
        )
        if temp_dir is not None:
            shutil.rmtree(temp_dir)
        return executable

    def generate_module(self) -> RawMlirModule:
        # xdsl_func input must be read only, clone it first
        xdsl_func = self._implementer.xdsl_func.clone()
        if self._implementer.no_alias:
            brand_inputs_with_noalias(xdsl_func)
        return MlirModule(xdsl_func)


class MlirModuleCompiler:
    def __init__(
        self,
        mlir_module: RawMlirModule,
        mlir_schedule: MlirSchedule | None = None,
        **kwargs: Any,
    ):
        self._mlir_module = mlir_module
        self._mlir_schedule = mlir_schedule
        self.mlir_install_dir = get_mlir_prefix(kwargs.get("mlir_install_dir", None))
        self.to_disassemble = kwargs.get("to_disassemble", "")
        self.save_temps = kwargs.get("save_temps", False)
        self.save_temps_dir = kwargs.get("save_temps_dir", "./save_temps_dir")
        self.bare_ptr = True
        self.print_source_ir = kwargs.get("print_source_ir", False)
        self.print_transformed_ir = kwargs.get("print_transformed_ir", False)
        self.print_assembly = kwargs.get("print_assembly", False)
        self.print_lowered_ir = kwargs.get("print_lowered_ir", False)
        self.debug = kwargs.get("debug", False)
        self.color = kwargs.get("color", False)
        self.shared_lib = kwargs.get("shared_lib", False)
        self.executable = kwargs.get("executable", False)
        self.dump_file = kwargs.get("dump_file")
        self.concluding_passes = kwargs.get("concluding_passes", [])
        self.always_vectorize = kwargs.get("always_vectorize", False)
        self.arch = kwargs.get("arch", "native")
        self.microarch = kwargs.get("microarch", "native")

    @property
    def cmd_cc(self):
        return [cc_bin]

    @property
    def cmd_opt(self):
        opt = [f"{self.mlir_install_dir}/bin/opt"]
        return opt + opt_opts + [f"-march={self.arch}", f"--mcpu={self.microarch}"]

    @property
    def cmd_llc(self):
        llc = [f"{self.mlir_install_dir}/bin/llc"]
        if self.arch == "native":
            llc_arch = [f"--mcpu={self.microarch}"]
        else:
            llc_arch = [f"-march={self.arch}", f"--mcpu={self.microarch}"]
        return llc + llc_opts + llc_arch

    @property
    def cmd_mlirtranslate(self):
        return [f"{self.mlir_install_dir}/bin/mlir-translate"] + mlirtranslate_opts

    @property
    def cmd_run_mlir(self):
        return [
            f"{self.mlir_install_dir}/bin/mlir-cpu-runner",
            *[f"-shared-libs={lib}" for lib in self.shared_libs],
        ] + mlirrunner_opts

    @property
    def shared_libs(self):
        return system_libs + [
            f"{self.mlir_install_dir}/lib/{lib}" for lib in runtime_libs
        ]

    @property
    def shared_path(self):
        return [f"-Wl,--rpath={self.mlir_install_dir}/lib/"]

    @property
    def disassemble_option(self):
        if not self.to_disassemble:
            return "--disassemble"
        else:
            return f"--disassemble={self.to_disassemble}"

    def build_disassemble_extra_opts(
        self,
        obj_file: str,
    ) -> list[str]:
        disassemble_extra_opts = [obj_file]
        if self.color:
            disassemble_extra_opts += objdump_color_opts
        return disassemble_extra_opts

    def build_run_extra_opts(self, obj_file: str) -> list[str]:
        run_extra_opts: list[str] = []
        if self.print_assembly:
            run_extra_opts += [
                "--dump-object-file",
                f"--object-filename={obj_file}",
            ]
        return run_extra_opts

    def dump_ir(self, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(self._mlir_module.mlir_module), file=sys.stderr)

    def mlir_transform(self) -> None:
        transform_op = [op for op in self._mlir_module.mlir_module.body.operations][-1]
        transform = isinstance(transform_op, NamedSequenceOp)
        if not transform:
            return
        pm = PassManager("builtin.module", context=self._mlir_module.mlir_context)  # type: ignore
        for opt in transform_opts:
            pm.add(opt)  # type: ignore
        pm.run(self._mlir_module.mlir_module)
        transform_op = [op for op in self._mlir_module.mlir_module.body.operations][-1]
        assert isinstance(transform_op, NamedSequenceOp)
        transform_op.erase()
        if self.print_transformed_ir:
            self.dump_ir("IR Dump After transform")

    def mlir_compile(self) -> None:
        if self.print_source_ir:
            self.dump_ir("IR Dump Before transform")
        self.mlir_transform()
        pm = PassManager("builtin.module", context=self._mlir_module.mlir_context)  # type: ignore
        for opt in lowering_opts:
            pm.add(opt)  # type: ignore
        pm.run(self._mlir_module.mlir_module)
        if self.print_lowered_ir:
            self.dump_ir("IR Dump After MLIR Opt")

    def disassemble(
        self,
        obj_file: str,
    ) -> subprocess.CompletedProcess:
        disassemble_extra_opts = self.build_disassemble_extra_opts(obj_file=obj_file)
        symbol = [f"{self.disassemble_option}"]
        objdump = objdump_arm_bin if self.arch == "aarch64" else objdump_bin
        disassemble_cmd = [objdump] + objdump_opts + symbol + disassemble_extra_opts
        print(" ".join(disassemble_cmd))
        dis_process = self.execute_command(cmd=disassemble_cmd, pipe_stdoutput=False)
        return dis_process

    def execute_command(
        self,
        cmd: list[str],
        input_pipe: str | None = None,
        pipe_stdoutput: bool = True,
    ) -> subprocess.CompletedProcess:
        pretty_cmd = "| " if input_pipe else ""
        pretty_cmd += " ".join(cmd)
        if self.debug:
            print(f"> exec: {pretty_cmd}", file=sys.stderr)

        if input_pipe and pipe_stdoutput:
            result = subprocess.run(
                cmd, input=input_pipe, stdout=subprocess.PIPE, text=True
            )
        elif input_pipe and not pipe_stdoutput:
            result = subprocess.run(cmd, input=input_pipe, text=True)
        elif not input_pipe and pipe_stdoutput:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        else:
            result = subprocess.run(cmd, text=True)
        return result

    def evaluate(self) -> str:
        self.schedule_module()
        self.mlir_compile()

        obj_dump_file = f"{self.dump_file}.o"
        run_extra_opts = self.build_run_extra_opts(
            obj_file=obj_dump_file,
        )
        cmd_run = self.cmd_run_mlir + run_extra_opts
        result = self.execute_command(
            cmd=cmd_run, input_pipe=str(self._mlir_module.mlir_module)
        )
        if self.print_assembly:
            disassemble_process = self.disassemble(
                obj_file=obj_dump_file,
            )
            assert disassemble_process.returncode == 0
        return result.stdout

    def schedule_module(self) -> None:
        if self._mlir_schedule is not None:
            scheduler = MlirModuleScheduler(
                self._mlir_module,
                self._mlir_schedule.schedule_impl,
                concluding_passes=self.concluding_passes,
                always_vectorize=self.always_vectorize,
            )
            scheduler.implement()

    def generate_without_compilation(self) -> str:
        self.schedule_module()
        return str(self._mlir_module)

    def _save_temp(self, fname: str, content: Any) -> None:
        if not self.save_temps:
            return
        os.makedirs(self.save_temps_dir, exist_ok=True)
        with open(f"{self.save_temps_dir}/{fname}", "w") as outf:
            outf.write(str(content))

    def compile(self) -> None:
        save_temp = self._save_temp
        save_temps_dir = self.save_temps_dir
        dump_file = self.dump_file
        temp_dir = None
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/payload"
        if self.save_temps:
            assert self.dump_file is not None, "TODO: save_temp requires dump_file"
            dump_tmp_dir = save_temps_dir
            os.makedirs(save_temps_dir, exist_ok=True)
        else:
            dump_tmp_dir = Path(dump_file).parent

        dump_base = Path(dump_file).name
        dump_tmp_file = f"{dump_tmp_dir}/{dump_base}"
        ir_dump_file = f"{dump_tmp_file}.ir"
        bc_dump_file = f"{dump_tmp_file}.bc"
        obj_dump_file = f"{dump_tmp_file}.o"
        exe_c_file = f"{dump_tmp_file}.main.c"
        so_dump_file = f"{dump_file}.so"
        exe_dump_file = f"{dump_file}.out"
        src_ir_dump_file = f"{dump_base}.mlir"
        mlir_llvm_dump_file = f"{dump_base}.llvm.mlir"

        self.schedule_module()
        save_temp(src_ir_dump_file, self._mlir_module.mlir_module)

        self.mlir_compile()
        save_temp(mlir_llvm_dump_file, self._mlir_module.mlir_module)

        translate_cmd = self.cmd_mlirtranslate + ["-o", ir_dump_file]
        llvmir_process = self.execute_command(
            cmd=translate_cmd,
            input_pipe=str(self._mlir_module.mlir_module),
        )
        assert llvmir_process.returncode == 0

        opt_pic = ["--relocation-model=pic"] if self.shared_lib else []
        opt_cmd = self.cmd_opt + opt_pic + [ir_dump_file, "-o", bc_dump_file]
        opt_process = self.execute_command(cmd=opt_cmd)
        assert opt_process.returncode == 0

        llc_cmd = self.cmd_llc + opt_pic + [bc_dump_file, "-o", obj_dump_file]
        bc_process = self.execute_command(cmd=llc_cmd)
        assert bc_process.returncode == 0

        if self.print_assembly:
            disassemble_process = self.disassemble(obj_file=obj_dump_file)
            assert disassemble_process.returncode == 0

        payload_objs = [obj_dump_file, *self.shared_libs]
        payload_path = [*self.shared_path]
        if self.shared_lib:
            shared_cmd = [
                *self.cmd_cc,
                *shared_lib_opts,
                obj_dump_file,
                "-o",
                so_dump_file,
                *self.shared_libs,
                *self.shared_path,
            ]
            shlib_process = self.execute_command(cmd=shared_cmd)
            assert shlib_process.returncode == 0

            payload_objs = [so_dump_file]
            payload_path = ["-Wl,--rpath=${ORIGIN}"]

        if self.executable:
            exe_cmd = [
                *self.cmd_cc,
                *exe_opts,
                exe_c_file,
                "-o",
                exe_dump_file,
                *payload_objs,
                *payload_path,
            ]
            with open(exe_c_file, "w") as outf:
                outf.write("extern void entry(void); int main() { entry(); return 0; }")
            exe_process = self.execute_command(cmd=exe_cmd)
            assert exe_process.returncode == 0

        if not self.save_temps:
            Path(ir_dump_file).unlink(missing_ok=True)
            Path(bc_dump_file).unlink(missing_ok=True)
            Path(obj_dump_file).unlink(missing_ok=True)
            Path(exe_c_file).unlink(missing_ok=True)
        if temp_dir is not None:
            shutil.rmtree(temp_dir)


class MlirModuleScheduler:
    def __init__(
        self,
        module: RawMlirModule,
        schedule_impl: list[MlirNodeSchedule],
        concluding_passes: list[str] = [],
        always_vectorize: bool = True,
    ) -> None:
        self._module = module
        self._loc = Location.unknown(self._module.mlir_context)
        self._schedules = schedule_impl
        self._concluding_passes = concluding_passes
        self._always_vectorize = always_vectorize
        self.named_sequence: NamedSequenceOp | None = None

    def implement(self) -> None:
        with (
            InsertionPoint(self._module.mlir_module.body),
            self._module.mlir_context,
            self._loc,
        ):
            self._module.mlir_module.operation.attributes[
                "transform.with_named_sequence"
            ] = UnitAttr.get()
            self.named_sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )
        with (
            InsertionPoint.at_block_begin(self.named_sequence.body),
            self._module.mlir_context,
            self._loc,
        ):
            if len(self._schedules) > 0:
                self._implement()
            else:
                transform.YieldOp([])

    def _generate_vectorization(self, handle: OpResult) -> OpResult:
        if self._always_vectorize or self._needs_vectorization():
            handle = structured.VectorizeChildrenAndApplyPatternsOp(handle)
            with InsertionPoint(transform.ApplyPatternsOp(handle).patterns):
                vector.ApplyLowerOuterProductPatternsOp()
                vector.ApplyLowerContractionPatternsOp()
        return handle

    def _needs_vectorization(self) -> bool:
        for schedule in self._schedules:
            if self._node_needs_vectorization(schedule):
                return True
        return False

    def _generate_tiling(self) -> OpResult:
        assert self.named_sequence is not None
        handle = None
        for schedule in self._schedules:
            match0 = structured_match(
                results_=transform.AnyOpType.get(),
                target=self.named_sequence.bodyTarget,
                op_attrs={schedule.node_ident: UnitAttr.get()},
            )
            handle = self._generate_node_tiling(match0, schedule)
        assert handle, "At least 1 operation should have been processed"
        return handle

    def _generate_unroll(self, handle: OpResult) -> None:
        for schedule in self._schedules:
            self._generate_node_unroll(handle, schedule)

    def _implement(self) -> None:
        assert self.named_sequence is not None
        with (
            InsertionPoint.at_block_begin(self.named_sequence.body),
            self._module.mlir_context,
            self._loc,
        ):
            handle = self._generate_tiling()
            handle = get_parent_op(
                transform.AnyOpType.get(),
                handle,
                isolated_from_above=True,
            )
            handle = self._generate_vectorization(handle)
            self._generate_unroll(handle)
            for p in self._concluding_passes:
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(), handle, pass_name=p
                )
            transform.YieldOp([])

    def _node_needs_vectorization(self, schedule: MlirNodeSchedule) -> bool:
        return len(schedule.vectorization) > 0

    def _generate_node_tiling(
        self, handle: OpResult, schedule: MlirNodeSchedule
    ) -> OpResult:
        # Produce the sequence of commands needed for the tiling
        tiling_arrays: dict[str, list[int]] = {}
        deepest_tiling = max(schedule.tiles.values(), key=len)
        depth_deepest_tiling = len(deepest_tiling)
        for tile_level in range(depth_deepest_tiling):
            for index_of_dim, (_, tiles) in enumerate(schedule.tiles.items()):
                # This dimension is not tiled at this level.
                if tile_level >= len(tiles):
                    continue

                # Create the array describing the tiling of this
                # dimension. If I have a (x,y,z) nest and I want
                # to tile the y dimension with a tile size of 16,
                # the resulting array is [0,16,0].
                tile_dim_name = list(tiles.keys())[tile_level]
                tiling_array = [
                    tiles[tile_dim_name] if i == index_of_dim else 0
                    for i in range(len(schedule.tiles))
                ]
                tiling_arrays[tile_dim_name] = tiling_array
        # Reorder the tiling according to permutation.
        tiling_arrays = {p: tiling_arrays[p] for p in schedule.permutation}
        # Materialize loops
        op_to_tile = handle
        all_loops = []
        for tile_name, tiling_array in tiling_arrays.items():
            # Useless to materialize a loop which will be vectorized
            if tile_name in schedule.vectorization:
                break
            # Generate the tiling itself
            if tile_name in schedule.parallelization:
                tiling_command = structured.TileUsingForallOp(
                    op_to_tile, tile_sizes=tiling_array
                )
            else:
                tiling_command = structured.TileUsingForOp(
                    op_to_tile, sizes=tiling_array
                )
            # Annotate the resulting loop if successfully generated
            if len(tiling_command.results) > 1:
                generated_loop = tiling_command.results[1]
                transform.AnnotateOp(
                    generated_loop, f"{schedule.node_ident}{tile_name}"
                )
                all_loops.append(generated_loop)
            #
            op_to_tile = tiling_command.results[0]

        # Stamp the outermost loop
        outer_loop = all_loops[0]
        for s in schedule.loop_stamps:
            transform.AnnotateOp(outer_loop, s)

        return outer_loop

    def _generate_node_unroll(
        self, handle: OpResult, schedule: MlirNodeSchedule
    ) -> None:
        for dim, factor in schedule.unrolling.items():
            match0 = structured_match(
                results_=transform.AnyOpType.get(),
                target=handle,
                op_attrs={f"{schedule.node_ident}{dim}": UnitAttr.get()},
            )
            # TODO: LLVM metadata instead of transform unroll may put less pressure
            # on MLIR front-end
            # https://llvm.org/docs/LangRef.html#llvm-loop
            loop_unroll(match0, factor)
