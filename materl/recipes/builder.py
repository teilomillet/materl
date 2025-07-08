"""
materl.recipes.builder

Defines the GraphBuilder class used by recipes to construct a symbolic graph.
"""
from typing import Dict, Optional, TYPE_CHECKING
from ..graph import Graph, SymbolicNode

if TYPE_CHECKING:
    from ..agents import Agent

class GraphBuilder:
    """
    A stateful builder that constructs a symbolic graph.
    This class is passed to recipe functions to provide a clean API
    for defining the computational graph.
    """
    def __init__(self, name: str):
        self.graph = Graph(name)
        self.graph.builder = self  # Attach builder to the graph
        self.node_map: Dict[str, SymbolicNode] = {}
        self.loss_node: Optional[SymbolicNode] = None

    def _add_node(self, op: str, **kwargs) -> SymbolicNode:
        """
        Helper to create and track a new node. This is the core logic that
        translates a declarative call into a graph component.
        """
        # Dependencies are passed as keyword arguments. We extract them to build
        # the dependency list for the new node.
        dependencies = [v for v in kwargs.values() if isinstance(v, SymbolicNode)]
        
        # The name of the node is also passed via kwargs, defaulting to the op name.
        # We use .get() instead of .pop() to ensure 'name' remains in op_kwargs
        # so the compiler can pass it to the underlying function.
        name = kwargs.get("name", op)
        
        # 1. Create the SymbolicNode instance.
        node = SymbolicNode(op=op, name=name, op_kwargs=kwargs, dependencies=dependencies)
        
        # 2. Add the created node to the graph, which now matches the expected
        # signature of `Graph.add_node`.
        self.graph.add_node(node)
        
        self.node_map[name] = node
        return node

    # Explicit Symbolic Operations
    def generate(self, **kwargs) -> SymbolicNode:
        return self._add_node("generate", **kwargs)

    def logprobs(self, **kwargs) -> SymbolicNode:
        # Special handling for 'is_ref' to create a unique node name for the
        # reference model's logprobs, avoiding name clashes in the graph.
        if kwargs.get("is_ref"):
            kwargs.setdefault("name", "logprobs_2")
        return self._add_node("logprobs", **kwargs)

    def advantages(self, **kwargs) -> SymbolicNode:
        return self._add_node("advantages", **kwargs)

    def returns(self, **kwargs) -> SymbolicNode:
        """
        Adds a symbolic node for computing discounted returns to the graph.
        This is a key operation for the REINFORCE algorithm.
        """
        return self._add_node("returns", **kwargs)

    def values(self, **kwargs) -> SymbolicNode:
        return self._add_node("values", **kwargs)

    def reward(self, **kwargs) -> SymbolicNode:
        return self._add_node("reward", **kwargs)

    def set_loss(self, **kwargs):
        """A special method to define the final loss node."""
        self.loss_node = self._add_node("loss", name="loss", **kwargs)

    def get_graph(self) -> Graph:
        """Returns the completed graph."""
        if self.loss_node is None:
            raise ValueError("A loss node must be set before getting the graph.")
        return self.graph 