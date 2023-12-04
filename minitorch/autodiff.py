from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    diff1 = [value if x != arg else value - epsilon for x, value in enumerate(vals)]
    diff2 = [value if x != arg else value + epsilon for x, value in enumerate(vals)]
    return (f(*diff2)-f(*diff1))/(2*epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # ~TODO: Implement for Task 1.4.
    # using dfs
    visited = []
    sorted_vars = []
    def dfs_visit(var:Variable):
        if var.is_constant() or var.unique_id in visited:
            return
        if not var.is_leaf():
            for i in var.parents: # var.history.inputs
                dfs_visit(i)
        visited.append(var.unique_id)
        sorted_vars.append(var)
    dfs_visit(variable)
    return sorted_vars

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    order_vars = topological_sort(variable)
    derivative_nodes = {variable.unique_id: deriv} # use a dictionary to store its "parent"
    for var in order_vars:
        if var.is_leaf():
            continue
        if var.unique_id in derivative_nodes.keys():
            deriv = derivative_nodes[var.unique_id]

        for curr_var, curr_deriv in var.chain_rule(deriv):
            if curr_var.is_leaf():
                curr_var.accumulate_derivative(curr_deriv)
            elif curr_var.unique_id in derivative_nodes.keys():
                derivative_nodes[curr_var.unique_id] += curr_deriv
            else:
                derivative_nodes[curr_var.unique_id] = curr_deriv  # add it to the dict


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
