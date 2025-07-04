from typing import Dict, List, Optional, Union
import torch
import einops


def _product(sequence: List[int], permutation_axes: List[int]) -> int:
    """minimalistic product that works both with numbers and symbols. Supports empty lists"""
    if not sequence:
        return "1"
    result = ""
    for element in sequence:
        result += f"tensor.shape[{permutation_axes.index(element)}] * "
    result = result.rstrip(" * ")
    return result


def _reconstruct_from_shape_uncached(
    self: einops.einops.TransformRecipe,
    shape: List[int],
    axes_dims: einops.einops.FakeHashableAxesLengths,
) -> einops.einops.CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, theano) and UnknownSize (tf, previously mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    # magic number
    need_init_reshape = False

    # last axis is allocated for collapsed ellipsis
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    for axis, dim in axes_dims:
        axes_lengths[self.axis_name2elementary_axis[axis]] = dim

    for input_axis, (known_axes, unknown_axes) in enumerate(
        self.input_composition_known_unknown
    ):
        length = shape[input_axis]
        if len(known_axes) == 0 and len(unknown_axes) == 1:
            # shortcut for the most common case
            axes_lengths[unknown_axes[0]] = length
            continue

        known_product = 1
        for axis in known_axes:
            known_product *= axes_lengths[axis]

        if len(unknown_axes) == 0:
            if (
                isinstance(length, int)
                and isinstance(known_product, int)
                and length != known_product
            ):
                raise einops.einops.EinopsError(
                    f"Shape mismatch, {length} != {known_product}"
                )
        else:
            # assert len(unknown_axes) == 1, 'this is enforced when recipe is created, so commented out'
            if (
                isinstance(length, int)
                and isinstance(known_product, int)
                and length % known_product != 0
            ):
                raise einops.einops.EinopsError(
                    f"Shape mismatch, can't divide axis of length {length} in chunks of {known_product}"
                )

            unknown_axis = unknown_axes[0]
            inferred_length: int = length // known_product
            axes_lengths[unknown_axis] = inferred_length

        if len(known_axes) + len(unknown_axes) != 1:
            need_init_reshape = True

    # at this point all axes_lengths are computed (either have values or variables, but not Nones)

    # elementary axes are ordered as they appear in input, then all added axes
    init_shapes: Optional[List[int]] = (
        axes_lengths[: len(self.axes_permutation)] if need_init_reshape else None
    )

    need_final_reshape = False
    final_shapes: List[int] = []
    _permutation_axes = self.axes_permutation or list(range(len(axes_lengths)))
    for grouping in self.output_composite_axes:
        final_shapes.append(_product(grouping, _permutation_axes))
        if len(grouping) != 1:
            need_final_reshape = True
    if final_shapes:
        final_shapes = str(final_shapes).replace("'", "")

    added_axes: Dict[int, int] = {
        pos: axes_lengths[pos_in_elementary]
        for pos, pos_in_elementary in self.added_axes.items()
    }

    # this list can be empty
    reduced_axes = list(range(self.first_reduced_axis, len(self.axes_permutation)))

    n_axes_after_adding_axes = len(added_axes) + len(self.axes_permutation)

    axes_reordering: Optional[List[int]] = self.axes_permutation
    if self.axes_permutation == list(range(len(self.axes_permutation))):
        axes_reordering = None

    _final_shapes = final_shapes if need_final_reshape else None
    return (
        init_shapes,
        axes_reordering,
        reduced_axes,
        added_axes,
        _final_shapes,
        n_axes_after_adding_axes,
    )


def rearrange(
    tensor: Union[torch.Tensor, List[torch.Tensor]], pattern: str, **axes_lengths
) -> torch.Tensor:
    hashable_axes_lengths = tuple(axes_lengths.items())
    shape = tensor.shape
    recipe = einops.einops._prepare_transformation_recipe(
        pattern, "rearrange", axes_names=tuple(axes_lengths), ndim=len(shape)
    )
    _result = _reconstruct_from_shape_uncached(recipe, shape, hashable_axes_lengths)
    (
        init_shapes,
        axes_reordering,
        reduced_axes,
        added_axes,
        final_shapes,
        n_axes_w_added,
    ) = _result
    if init_shapes is not None:
        print(f"torch.reshape(tensor, {init_shapes})")
    if axes_reordering is not None:
        print(f"torch.permute(tensor, {axes_reordering})")
    if len(added_axes) > 0:
        repeats = [-1] * n_axes_w_added
        for axis_position, axis_length in added_axes.items():
            print(f"torch.unsqueeze(tensor, {axis_position})")
            repeats[axis_position] = axis_length
        print(f"torch.expand(tensor, {repeats})")
    if final_shapes is not None:
        print(f"torch.reshape(tensor, {final_shapes})")
