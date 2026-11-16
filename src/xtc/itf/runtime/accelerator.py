# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import Any, Callable
import ctypes

from xtc.itf.comp.module import Module
from xtc.itf.runtime.common import CommonRuntimeInterface


class AcceleratorDevice(CommonRuntimeInterface, ABC):
    """Abstract interface for an accelerator device (such as GPU, MPPA, etc)."""

    @abstractmethod
    def detect_accelerator(self) -> bool:
        """Detect if the accelerator device is available.

        Returns:
            A boolean representing if the accelerator device is available.
        """
        ...

    @abstractmethod
    def init_device(self) -> None:
        """Initialize the accelerator device.

        This method is called to initialize the accelerator device.
        """
        ...

    @abstractmethod
    def deinit_device(self) -> None:
        """Deinitialize the accelerator device.

        This method is called to deinitialize the accelerator device.
        """
        ...

    @abstractmethod
    def load_module(self, module: Module) -> None:
        """Load a module on the accelerator device.

        Args:
            module (AcceleratorModule): The module to load.
        """
        ...

    @abstractmethod
    def get_module_function(self, module: Module, function_name: str) -> Callable:
        """Get a function from a module on the accelerator device.

        Args:
            module (AcceleratorModule): The module to get the function from.
            function_name (str): The name of the function to get.
        """
        ...

    @abstractmethod
    def unload_module(self, module: Module) -> None:
        """Unload a module from the accelerator device.

        Args:
            module (AcceleratorModule): The module to unload.
        """
        ...

    @abstractmethod
    def memory_allocate(self, size_bytes: int, memory: str | None = None) -> Any:
        """Allocate memory on the accelerator device.

        Args:
            size_bytes (int): The size in bytes to allocate.
            memory (str | None): The device memory to allocate in. None or
                "global" (alias "ddr") is the device default memory.

        Returns:
            A handle or reference to the allocated memory.

        Raises:
            ValueError: if the memory is not supported by the device.
        """
        ...

    def supported_memories(self) -> tuple[str, ...]:
        """Canonical names of the memories supported by the device.

        Devices extend this with their specific memories.
        """
        return ("global",)

    def resolve_memory(self, memory: str | None) -> str:
        """Return the canonical memory name, "ddr" being an alias of "global".

        Raises:
            ValueError: if the memory is not supported by the device.
        """
        if memory is None or memory == "ddr":
            return "global"
        if memory not in self.supported_memories():
            raise ValueError(
                f"unsupported memory {memory!r} for {type(self).__name__}, "
                f"supported: {(*self.supported_memories(), 'ddr')}"
            )
        return memory

    @abstractmethod
    def memory_free(self, handle: Any) -> None:
        """Free memory on the accelerator device.

        Args:
            handle (Any): The handle to the memory to free.
        """
        ...

    @abstractmethod
    def memory_copy_to(
        self, acc_handle: Any, src: ctypes.c_void_p, size_bytes: int
    ) -> None:
        """Copy memory from the host to the accelerator device.

        Args:
            acc_handle (Any): The handle to the memory to copy to.
            src (ctypes.c_void_p): The source data pointer.
            size_bytes (int): The size in bytes to copy.
        """
        ...

    @abstractmethod
    def memory_copy_from(
        self, acc_handle: Any, dst: ctypes.c_void_p, size_bytes: int
    ) -> None:
        """Copy memory from the accelerator device to the host.

        Args:
            acc_handle (Any): The handle to the memory to copy from.
            dst (ctypes.c_void_p): The destination data pointer.
            size_bytes (int): The size in bytes to copy.
        """
        ...

    @abstractmethod
    def memory_fill_zero(self, acc_handle: Any, size_bytes: int) -> None:
        """Fill memory on the accelerator device with zeros.

        Args:
            acc_handle (Any): The handle to the memory to fill with zeros.
            size_bytes (int): The size in bytes to fill with zeros.
        """
        ...

    @abstractmethod
    def memory_data_pointer(self, acc_handle: Any) -> ctypes.c_void_p:
        """Get the data pointer of the memory on the accelerator device.

        Args:
            acc_handle (Any): The handle to the memory to get the data pointer of.
        """
        ...

    # TODO: describe hardware architecture
    # TODO: profiling and traces
