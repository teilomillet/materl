"""
materl.algorithm

This module defines the @algorithm decorator for transforming base recipes
into new algorithms through simple modifications.
"""
from functools import wraps

class GraphTransformer:
    """
    A simple interface for modifying graphs.
    This is what users get in their transformation functions.
    """
    def __init__(self, base_graph):
        self.graph = base_graph
        self.builder = base_graph.builder
    
    def get_node_by_name(self, name: str):
        """Finds a node in the graph by its name."""
        for node in self.graph.nodes:
            if node.name == name:
                return node
        raise ValueError(f"Node with name '{name}' not found in graph.")

    def replace_loss(self, loss_fn: str, **loss_kwargs):
        """Replace the loss function with a new one."""
        loss_node = self.get_node_by_name("loss")
        loss_node.op_kwargs["loss_fn"] = loss_fn
        loss_node.op_kwargs.update(loss_kwargs)
        
        # Also pass values if they now exist
        if "values" in self.builder.node_map:
            loss_node.op_kwargs["values"] = self.builder.node_map["values"]
            loss_node.dependencies.append(self.builder.node_map["values"])

        return self
    
    def replace_reward(self, reward_fn: str, **reward_kwargs):
        """Replace the reward function with a new one."""
        reward_node = self.get_node_by_name("reward")
        reward_node.op_kwargs["name"] = reward_fn
        reward_node.op_kwargs.update(reward_kwargs)
        return self

    def add_node(self, op: str, name: str, **op_kwargs):
        """Adds a new node to the graph using the original builder."""
        # Use the builder's method to correctly add the node
        new_node = self.builder._add_node(op, name=name, **op_kwargs)
        return new_node

    def add_values_node(self, value_model, enable_gae=True, enable_value_loss=True):
        """
        Adds a 'values' node to compute state values from a value model.
        
        Args:
            value_model: The model to use for value computation
            enable_gae: If True, modifies advantages to use GAE
            enable_value_loss: If True, adds value loss to the objective
        """
        completions = self.get_node_by_name("generate")
        # Use 'value_model' parameter name to match compute_values function signature
        values = self.add_node("values", name="values", value_model=value_model, completions=completions)
        
        if enable_gae:
            advantages_node = self.get_node_by_name("advantages")
            advantages_node.op_kwargs["values"] = values
            if values not in advantages_node.dependencies:
                advantages_node.dependencies.append(values)
        
        if enable_value_loss:
            loss_node = self.get_node_by_name("loss")
            loss_node.op_kwargs["values"] = values
            if values not in loss_node.dependencies:
                loss_node.dependencies.append(values)
        
        return values

def algorithm(using):
    """
    A decorator that transforms a base recipe into a new algorithm.
    
    Usage:
        @algorithm(using=grpo)
        def my_new_algo(graph, policy, ref_policy, prompts, max_completion_length):
            # Apply transformations to the base graph
            return graph.replace_loss("my_custom_loss", param=value)
    """
    base_recipe_func = using
    
    def decorator(transform_func):
        @wraps(transform_func)
        def wrapper(*args, **kwargs):
            import inspect
            
            # 1. Get the signature of the base recipe to filter arguments
            base_sig = inspect.signature(base_recipe_func)
            base_params = set(base_sig.parameters.keys())
            
            # 2. Filter kwargs to only include what the base recipe expects
            # Skip the first parameter which is the GraphBuilder ('ml')
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in base_params}
            
            # 3. Generate the base graph from the recipe
            base_graph = base_recipe_func(*args, **filtered_kwargs)
            
            # 4. Create a transformer interface
            transformer = GraphTransformer(base_graph)
            
            # 5. Apply the user's transformation with ALL original arguments
            result = transform_func(transformer, *args, **kwargs)
            
            # 6. Return the modified graph
            if result is None:
                return base_graph
            elif hasattr(result, 'graph'):
                return result.graph
            else:
                return result
        
        return wrapper
    return decorator 