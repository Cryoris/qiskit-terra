# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A ListOp that returns a sparse array upon eval if possible."""

from typing import Union, Optional, Dict
import numpy as np

from qiskit.quantum_info import Statevector

from .list_op import ListOp
from ..operator_base import OperatorBase


class SparseVectorOp(ListOp):
    """A ListOp that returns a sparse array upon eval if possible."""

    def eval(self,
             front: Optional[
                 Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]
             ] = None,
             ) -> Union[OperatorBase, complex]:
        """
        Evaluate the Operator's underlying function, either on a binary string or another Operator.
        A square binary Operator can be defined as a function taking a binary function to another
        binary function. This method returns the value of that function for a given StateFn or
        binary string. For example, ``op.eval('0110').eval('1110')`` can be seen as querying the
        Operator's matrix representation by row 6 and column 14, and will return the complex
        value at those "indices." Similarly for a StateFn, ``op.eval('1011')`` will return the
        complex value at row 11 of the vector representation of the StateFn, as all StateFns are
        defined to be evaluated from Zero implicitly (i.e. it is as if ``.eval('0000')`` is already
        called implicitly to always "indexing" from column 0).

        ListOp's eval recursively evaluates each Operator in ``oplist``,
        and combines the results using the recombination function ``combo_fn``.

        Args:
            front: The bitstring, dict of bitstrings (with values being coefficients), or
                StateFn to evaluated by the Operator's underlying function.

        Returns:
            The output of the ``oplist`` Operators' evaluation function, combined with the
            ``combo_fn``. If either self or front contain proper ``ListOps`` (not ListOp
            subclasses), the result is an n-dimensional list of complex or StateFn results,
            resulting from the recursive evaluation by each OperatorBase in the ListOps.

        Raises:
            NotImplementedError: Raised if called for a subclass which is not distributive.
            TypeError: Operators with mixed hierarchies, such as a ListOp containing both
                PrimitiveOps and ListOps, are not supported.
            NotImplementedError: Attempting to call ListOp's eval from a non-distributive subclass.

        """
        # pylint: disable=cyclic-import
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.vector_state_fn import VectorStateFn

        # The below code only works for distributive ListOps, e.g. ListOp and SummedOp
        if not self.distributive:
            raise NotImplementedError("ListOp's eval function is only defined for distributive "
                                      "ListOps.")

        evals = []
        for op in self.oplist:
            if isinstance(op, DictStateFn) and front is None:
                evals.append(op.to_spmatrix())
            else:
                evals.append(op.eval(front))

        # Handle application of combo_fn for DictStateFn resp VectorStateFn operators
        if self._combo_fn != ListOp([])._combo_fn:
            if all(isinstance(op, DictStateFn) for op in evals) or \
                    all(isinstance(op, VectorStateFn) for op in evals):
                if not all(
                        op.is_measurement == evals[0].is_measurement for op in evals
                ):
                    raise NotImplementedError("Combo_fn not yet supported for mixed measurement "
                                              "and non-measurement StateFns")
                result = self.combo_fn(evals)
                if isinstance(result, list):
                    multiplied = self.coeff * np.array(result)
                    return multiplied.tolist()
                return self.coeff * result

        if all(isinstance(op, OperatorBase) for op in evals):
            return self.__class__(evals)  # type: ignore
        elif any(isinstance(op, OperatorBase) for op in evals):
            raise TypeError('Cannot handle mixed scalar and Operator eval results.')
        else:
            result = self.combo_fn(evals)
            if isinstance(result, list):
                multiplied = self.coeff * np.array(result)
                return multiplied.tolist()
            return self.coeff * result
