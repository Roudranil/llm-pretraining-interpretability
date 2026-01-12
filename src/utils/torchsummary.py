from typing import Literal, Optional

import torch
from torch import nn


def summary(
    model: nn.Module,
    input_size: Optional[torch.Tensor] = None,
    input_data: Optional[torch.Tensor] = None,
    device: Literal["cpu", "cuda"] = "cpu",
    depth: int = -1,
    precision: str = "32",
):
    """
    Concise tree + parameter stats for a PyTorch model.
    Pure PyTorch (no external deps).
    Adds output shape tracking with minimal changes.
    """
    if input_size is None and input_data is not None:
        input_size = input_data.shape  # type: ignore

    def get_precision_bytes(precision):
        """Get bytes per parameter based on precision"""
        precision = str(precision).lower()
        if precision in ["64", "64-true"]:
            return 8  # Double precision
        elif precision in ["32", "32-true"]:
            return 4  # Full precision
        elif precision in ["16", "16-mixed"]:
            return 2  # 16bit mixed precision
        elif precision in ["bf16", "bf16-mixed"]:
            return 2  # bfloat16 mixed precision
        else:
            return 4  # Default to full precision

    def get_max_depth(module, current_depth=0):
        """Calculate maximum depth of module tree"""
        children = list(module.named_children())
        if not children:
            return current_depth
        return max(get_max_depth(child, current_depth + 1) for _, child in children)

    def compute_effective_depth(depth, max_depth):
        """Convert negative depths to effective positive depths"""
        if depth >= 0:
            return depth
        else:
            # depth < 0 means full depth minus abs(depth + 1)
            return max(max_depth + depth + 1, 0)  # floor at 0

    def align_columns(
        lines, sep, names=["model", "#params", "w&b_shape", "in shape", "out shape"]
    ):
        """Align #params and shapes columns with optional in/out shapes"""
        parsed_lines = []
        max_cols = 0
        max_lens = [0] * len(names)  # max length for each column

        # Parse all lines and find max columns
        for line in lines:
            parts = line.split(sep)
            parts = [p.strip() for p in parts]
            max_cols = max(max_cols, len(parts))
            # Normalize to max columns
            while len(parts) < len(names):
                parts.append("")
            parsed_lines.append(parts)

            # Update max lengths for each column
            for i in range(min(len(parts), len(names))):
                max_lens[i] = max(max_lens[i], len(parts[i]))

        # Determine actual number of columns to use
        actual_cols = min(max_cols, len(names))

        # Build header using only the needed column names
        header_parts = [names[i].ljust(max_lens[i]) for i in range(actual_cols)]
        header = "  ".join(header_parts)

        # Build separator line
        separator_parts = ["=" * max_lens[i] for i in range(actual_cols)]
        separator = "  ".join(separator_parts)

        aligned_lines = [header, separator]

        # Build aligned data lines
        for parts in parsed_lines:
            line_parts = [parts[i].ljust(max_lens[i]) for i in range(actual_cols)]
            line = "  ".join(line_parts)
            aligned_lines.append(line)

        return aligned_lines

    def pretty_size(bytes_num):
        """Convert bytes to human readable format (B/KB/MB/GB/TB)"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_num < 1024 or unit == "TB":
                return f"{bytes_num:.2f} {unit}"
            bytes_num /= 1024

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def estimate_memory_footprint(
        model,
        input_data=None,
        activation_sizes=[],
        forward_pass_successful=False,
        precision: str = "32-true",
    ) -> None:
        """
        Estimates and prints memory usage for a PyTorch model.

        Args:
            model: PyTorch model
            input_data: Sample input tensor (optional)
            device: Device to move model and input_data to
            precision: Precision type - 64/'64'/'64-true', 32/'32'/'32-true',
                    16/'16'/'16-mixed', 'bf16'/'bf16-mixed'
        """
        bytes_per_param = get_precision_bytes(precision)  # Default to float32

        # Calculate parameter memory
        total_params, trainable_params = count_params(model)
        param_bytes = total_params * bytes_per_param
        optimizer_bytes = param_bytes * 3  # AdamW uses ~3x param memory

        # If no input data provided, print basic stats only
        if input_data is None:
            print(f"Params size: {pretty_size(param_bytes)}")
            print(f"Optimizer states (AdamW): {pretty_size(optimizer_bytes)}")
            return

        # Input size (always available)
        input_bytes = input_data.numel() * bytes_per_param
        if forward_pass_successful:
            # Calculate activation memory from actual forward pass
            total_activations = sum(activation_sizes)
            activations_bytes = total_activations * bytes_per_param
            # Forward + backward pass: activations + gradients + 20% peak buffer
            forward_pass_bytes = activations_bytes * 2 * 1.2
        else:
            # Fallback estimation: heuristic-based calculation
            # Forward pass creates activations ≈ parameter count in elements
            # Backward pass stores gradients of same size as activations
            # Total = 2 × activations + input tensor

            forward_pass_bytes = (total_params * bytes_per_param * 2) + (
                input_data.numel() * bytes_per_param
            )

        # Total memory
        total_bytes = input_bytes + forward_pass_bytes + param_bytes + optimizer_bytes

        # Print detailed statistics
        print(f"Input size: {pretty_size(input_bytes)}")
        print(f"Params size: {pretty_size(param_bytes)}")
        print(f"Forward/backward pass size: {pretty_size(forward_pass_bytes)}")
        print(f"Optimizer states (AdamW): {pretty_size(optimizer_bytes)}")
        print(f"Estimated Total Size: {pretty_size(total_bytes)}")
        if not forward_pass_successful:
            print(
                "\nNote: An exception occured when trying to forward input_data through model. Hence approximate fallback calculations used."
            )

    def get_module_info(module):
        # get weight and bias tensor shapes too
        class_name = None
        weight_shape = None
        bias_shape = None
        if hasattr(module, "weight"):
            try:
                weight_shape = tuple(module.weight.shape)
            except Exception as e:
                weight_shape = None
        if hasattr(module, "bias"):
            try:
                bias_shape = tuple(module.bias.shape)
            except Exception as e:
                bias_shape = None
        # Get constructor args from repr and clean it up
        repr_str = module.__repr__()
        if "(" in repr_str:
            class_name = repr_str.split("(")[0]
            # # Extract parameters from repr
            # if "\n" in repr_str:
            #     # Multi-line repr - get the parameters
            #     lines = repr_str.split("\n")
            #     params = []
            #     for line in lines[1:-1]:  # Skip first and last line
            #         line = line.strip()
            #         if ":" not in line and "=" in line:  # Constructor parameters
            #             params.append(line)
            #     if params:
            #         param_str = ", ".join(params)
            #         result = f"{class_name}({param_str})"
            #         if result.endswith("("):
            #             result = result[:-1]
            #         return result, weight_shape, bias_shape
            # first_line = repr_str.split("\n")[0]
            # if first_line.endswith("("):
            #     first_line = first_line[:-1]
            # return first_line, weight_shape, bias_shape
            # # return class_name
        # return repr_str, weight_shape, bias_shape
        return class_name, weight_shape, bias_shape

    def hook_fn(module, in_data, out_data):
        if isinstance(in_data, (tuple, list)):
            try:
                shapes = [
                    tuple(o.shape) if hasattr(o, "shape") else str(type(o))
                    for o in in_data
                ]
                input_shapes[module] = shapes if len(shapes) > 1 else shapes[0]
            except:
                input_shapes[module] = ""
        else:
            input_shapes[module] = (
                tuple(in_data.shape) if hasattr(in_data, "shape") else ""
            )
        if isinstance(out_data, (tuple, list)):
            try:
                shapes = [
                    tuple(o.shape) if hasattr(o, "shape") else str(type(o))
                    for o in out_data
                ]
                output_shapes[module] = shapes if len(shapes) > 1 else shapes[0]
            except:
                output_shapes[module] = ""
            try:
                for o in out_data:
                    if hasattr(o, "numel"):
                        activation_sizes.append(o.numel())
            except:
                pass
        else:
            output_shapes[module] = (
                tuple(out_data.shape) if hasattr(out_data, "shape") else ""
            )
            try:
                if hasattr(out_data, "numel"):
                    activation_sizes.append(out_data.numel())
            except:
                pass

    def _print(
        module,
        prefix="",
        is_root=False,
        print_func=print,
        sep="  \t ",
        depth=-1,
        current_depth=1,
    ):
        if depth == 0:
            return
        children = list(module.named_children())
        current_depth_index = 1
        for i, (name, child) in enumerate(children):
            index_str = f"[{current_depth} - {current_depth_index}]"
            current_depth_index += 1
            is_last = i == len(children) - 1
            branch = "└─ " if is_last else "├─ "
            total, trainable = count_params(child)
            # Check if this child has repeated modules in its repr
            repeated_info = ""
            child_repr = child.__repr__()
            if "\n" in child_repr:
                lines = child_repr.split("\n")
                for line in lines:
                    if "x " in line and "(" in line and ")" in line and ":" in line:
                        # Extract the "(0-3): 4 x ModuleName" part
                        repeated_part = line.strip(" ").rstrip("(")
                        if "x " in repeated_part:
                            repeated_info = f" [{repeated_part}]"
                            break
            # Get detailed module info
            module_info, weight_shape, bias_shape = get_module_info(child)

            # Add shape info if available
            shape_info = f"{sep}"
            wandb_shape_infos = []
            if weight_shape is not None:
                wandb_shape_infos.append(f"w: {weight_shape}")
            if bias_shape is not None:
                wandb_shape_infos.append(f"b: {bias_shape}")
            if len(wandb_shape_infos) > 0:
                shape_info += ", ".join(wandb_shape_infos)
            else:
                shape_info += "  "
            if child in output_shapes:
                # shape = output_shapes[child]
                shape_info += f"{sep}{input_shapes[child]}{sep}{output_shapes[child]}"
            line = f"{prefix}{branch}{name}: {module_info}{repeated_info} {index_str}"
            if total > 0:
                line += f"{sep}{total:,}"
            else:
                line += f"{sep} "
            line += shape_info
            print_func(line)
            extension = "    " if is_last else "│   "
            _print(
                child,
                prefix + extension,
                print_func=print_func,
                sep=sep,
                depth=(depth - 1 if depth > 0 else -1),
                current_depth=current_depth + 1,
            )

    total_params, trainable_params = count_params(model)

    # Store output shapes by module
    output_shapes = {}
    input_shapes = {}
    # get activations
    activation_sizes = []

    # Register hooks on leaf modules only
    hooks = []
    for mod in model.modules():
        # if mod != model:
        hooks.append(mod.register_forward_hook(hook_fn))

    # Run forward pass if input_size provided
    if input is not None:
        import torch

        model.eval()
        device_obj = torch.device(device)
        model.to(device_obj)
        # dummy_input = torch.zeros((1, *input_size), device=device_obj)
        try:
            with torch.no_grad():
                _ = model(input_data.to(device_obj))  # type: ignore
                forward_pass_successful = True
        except Exception as e:
            print(e)
            forward_pass_successful = False

    # Calculate model size estimates
    param_bytes = total_params * 4
    # Root
    # Collect all lines first for alignment
    lines = []

    def capture_print(line):
        lines.append(line)

    # Root
    sep = "  \t "
    root_info = get_module_info(model)
    lines.append(f"{root_info} | {total_params:,} params")
    max_depth = get_max_depth(model)
    effective_depth = compute_effective_depth(depth, max_depth)
    _print(
        model,
        is_root=True,
        print_func=capture_print,
        sep=sep,
        depth=effective_depth,
        current_depth=1,
    )

    # Align and print
    for line in align_columns(lines, sep=sep):
        print(line)

    print("\nParameter count statistics:\n==========================")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("\nMemory footprint statistcs:\n==========================")
    estimate_memory_footprint(
        model, input_data, activation_sizes, forward_pass_successful, precision
    )

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    model.to("cpu")
    model.to("cpu")
