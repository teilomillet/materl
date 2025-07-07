"""
materl.graph

This module defines the core symbolic components of the declarative API.

- Graph: A collection of SymbolicNodes that represents a full algorithm.
- SymbolicNode: A representation of a single operation in the graph.
"""
from typing import Any, Dict, List, Optional

class SymbolicNode:
    """Represents a single, symbolic operation in the computation graph."""
    def __init__(self, op: Any, name: Optional[str] = None, op_kwargs: Optional[Dict] = None, **kwargs):
        self.op = op
        self.name = name or str(op)
        self.op_kwargs = op_kwargs or {}
        
        # This will hold the other SymbolicNode objects that this node depends on.
        self.dependencies: List[SymbolicNode] = []
        for key, value in kwargs.items():
            if isinstance(value, SymbolicNode):
                self.dependencies.append(value)
            # We can also handle lists of nodes for operations that take multiple inputs
            elif isinstance(value, list) and all(isinstance(v, SymbolicNode) for v in value):
                self.dependencies.extend(value)

    def __repr__(self):
        return f"SymbolicNode(op={self.op}, name='{self.name}')"

    def __add__(self, other):
        """Allows composing nodes with the '+' operator, useful for rewards."""
        if not isinstance(other, SymbolicNode):
            raise TypeError("Can only compose SymbolicNode with another SymbolicNode.")
        
        # Create a new composite node representing the addition.
        return SymbolicNode(
            op="add", 
            name=f"({self.name} + {other.name})",
            input_nodes=[self, other]
        )


class Graph:
    """Represents a full algorithm as a directed acyclic graph of SymbolicNodes."""
    def __init__(self, name=""):
        self.name = name
        self.nodes: List[SymbolicNode] = []
        self.named_nodes: Dict[str, SymbolicNode] = {}

    def add_node(self, node: SymbolicNode):
        """Adds a new symbolic operation to the graph."""
        self.nodes.append(node)
        if node.name:
            if node.name in self.named_nodes:
                # Handle simple name conflicts by appending a number
                base_name = node.name
                i = 2
                while f"{base_name}_{i}" in self.named_nodes:
                    i += 1
                node.name = f"{base_name}_{i}"
            
            self.named_nodes[node.name] = node
            # Allow accessing the node via attribute, e.g., graph.advantages
            # Only set attribute if it's a valid identifier
            if node.name.isidentifier():
                setattr(self, node.name, node)
        return node

    def __repr__(self):
        return f"Graph(name='{self.name}', nodes={len(self.nodes)})" 