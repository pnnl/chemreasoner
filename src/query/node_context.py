"""
This module contains the NodeContext class for managing and processing JSON data
related to catalyst recommendations for chemical reactions.

Author: Rounak Meyur
"""

import json
import re
from typing import List, Dict, Any, Optional
from logging_utils import LogManager

# Load the logging configuration
if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_node_context.log", 
        log_config_path="src/query/logging_config.ini"
        )

logger = LogManager.get_logger("node_context")

class NodeContext:
    """
    A class for managing and processing JSON data related to catalyst recommendations.

    This class provides methods for loading JSON data, extracting specific usage information,
    and retrieving relevant nodes and context based on various criteria.

    Attributes:
        json_data (Dict[str, Any]): The loaded JSON data containing catalyst information.
        specific_usage (str): The specific usage extracted from the JSON data.
    """

    def __init__(self, json_file_path: str):
        """
        Initialize the NodeContext with the given JSON file path.

        Args:
            json_file_path (str): The path to the JSON file containing catalyst data.
        """
        self.json_data = self.load_json(json_file_path)
        self.specific_usage = self.extract_specific_usage()

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load and return the JSON data from the specified file path.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            Dict[str, Any]: The loaded JSON data as a dictionary.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    def extract_specific_usage(self) -> str:
        """
        Extract the specific usage information from the JSON data's template field.

        Returns:
            str: The extracted specific usage or a default value if not found.
        """
        template = self.json_data.get('template', '')
        match = re.search(r'Generate a list of top-5 .+ for (.+)\.', template)
        if match:
            return match.group(1)
        return "CO2 hydrogenation reaction to methanol"  # Default value if not found

    def get_nodes(self, scope_type: str, filter_options: Optional[List[str]] = None) -> List[int]:
        """
        Retrieve node IDs based on the specified scope type and filter options.

        Args:
            scope_type (str): The type of scope ('best_path' or 'all_nodes').
            filter_options (Optional[List[str]]): List of catalyst systems or elements to filter by.

        Returns:
            List[int]: A list of node IDs matching the criteria.
        """
        if scope_type == "best_path":
            node_ids = self.get_best_path_node_ids()
        else:
            node_ids = self.get_all_node_ids()
        
        if filter_options:
            return self.filter_nodes(node_ids, filter_options)
        return node_ids

    def filter_nodes(self, node_ids: List[int], filter_options: List[str]) -> List[int]:
        """
        Filter the given node IDs based on the provided filter options.

        Args:
            node_ids (List[int]): The list of node IDs to filter.
            filter_options (List[str]): The list of filter options to apply.

        Returns:
            List[int]: A list of filtered node IDs.
        """
        filtered_ids = []
        for node_id in node_ids:
            node = self.find_node_by_id(node_id)
            if node:
                answer = node['info']['generation'][0]['answer']
                if self.contains_exact_match(answer, filter_options):
                    filtered_ids.append(node_id)
        
        return filtered_ids if filtered_ids else node_ids

    def contains_exact_match(self, text: str, options: List[str]) -> bool:
        """
        Check if the text contains an exact match for any of the given options.

        Args:
            text (str): The text to search in.
            options (List[str]): The list of options to search for.

        Returns:
            bool: True if an exact match is found, False otherwise.
        """
        for option in options:
            pattern = r'\b' + re.escape(option) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def get_best_path_node_ids(self) -> List[int]:
        """
        Get the list of node IDs representing the best path in the decision tree.

        Returns:
            List[int]: A list of node IDs for the best path.
        """
        best_node = self.find_best_node(self.json_data)
        if best_node is None:
            return []
        return self.trace_path_to_root(best_node)

    def find_best_node(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Recursively find the node with the highest node_rewards in the tree.

        Args:
            node (Dict[str, Any]): The current node to examine.

        Returns:
            Optional[Dict[str, Any]]: The node with the highest node_rewards, or None if not found.
        """
        best_node = node
        if 'children' in node:
            for child in node['children']:
                child_best = self.find_best_node(child)
                if child_best['node_rewards'] > best_node['node_rewards']:
                    best_node = child_best
        return best_node

    def trace_path_to_root(self, node: Dict[str, Any]) -> List[int]:
        """
        Trace the path from the given node to the root node (id 0).

        Args:
            node (Dict[str, Any]): The starting node.

        Returns:
            List[int]: A list of node IDs representing the path to the root.
        """
        path = []
        current = node
        while current['id'] != 0:  # Continue until we reach the root node with id 0
            path.append(current['id'])
            current = self.find_parent(self.json_data, current['id'])
            if current is None:
                break  # This should not happen in a well-formed tree
        path.append(0)  # Add the root node id
        return list(reversed(path))

    def find_parent(self, node: Dict[str, Any], target_id: int) -> Optional[Dict[str, Any]]:
        """
        Find the parent node of the node with the given target_id.

        Args:
            node (Dict[str, Any]): The current node to examine.
            target_id (int): The ID of the node whose parent we're looking for.

        Returns:
            Optional[Dict[str, Any]]: The parent node, or None if not found.
        """
        if 'children' in node:
            for child in node['children']:
                if child['id'] == target_id:
                    return node
                parent = self.find_parent(child, target_id)
                if parent:
                    return parent
        return None

    def get_all_node_ids(self) -> List[int]:
        """
        Get a list of all node IDs in the JSON data structure.

        Returns:
            List[int]: A list of all node IDs.
        """
        all_node_ids = []
        queue = [self.json_data]
        while queue:
            node = queue.pop(0)
            all_node_ids.append(node['id'])
            queue.extend(node.get('children', []))
        return all_node_ids

    def get_context(self, node_ids: List[int], context_type: str) -> str:
        """
        Retrieve the context information for the given node IDs and context type.

        Args:
            node_ids (List[int]): The list of node IDs to get context for.
            context_type (str): The type of context to retrieve.

        Returns:
            Union[str, List[Dict[str, Any]]]: The context information.

        Raises:
            ValueError: If an unknown context type is provided.
        """
        if context_type == "catalyst_recommendation":
            return self.get_catalyst_recommendation_context(node_ids)
        elif context_type == "adsorption_energies":
            return self.get_adsorption_energies_context(node_ids)
        elif context_type == "reaction_pathways":
            return self.get_reaction_pathways_context(node_ids)
        else:
            raise ValueError(f"Unknown context type: {context_type}")

    def find_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Find and return the node with the given node_id.

        Args:
            node_id (int): The ID of the node to find.

        Returns:
            Optional[Dict[str, Any]]: The found node, or None if not found.
        """
        def dfs(node):
            if node['id'] == node_id:
                return node
            for child in node.get('children', []):
                result = dfs(child)
                if result:
                    return result
            return None
        
        return dfs(self.json_data)

    def get_catalyst_recommendation_context(self, node_ids: List[int]) -> str:
        """
        Generate a string containing catalyst recommendation context for the given node IDs.

        Args:
            node_ids (List[int]): The list of node IDs to get context for.

        Returns:
            str: A formatted string containing the catalyst recommendation context.
        """
        context_parts = ["Catalyst Recommendation Context:\n"]
        context_parts.append(f"This context provides information about catalyst recommendations for {self.specific_usage}.\n")
        context_parts.append("It includes node IDs, corresponding answers, and node rewards for each step in the decision path.\n")

        for node_id in node_ids:
            node = self.find_node_by_id(node_id)
            if node:
                context_parts.append(f"Node ID: {node['id']}")
                context_parts.append(f"Answer: {node['info']['generation'][0]['answer']}")
                context_parts.append(f"Node Reward: {node['node_rewards']}")
                context_parts.append("")  # Empty line for separation

        return "\n".join(context_parts)

    def get_adsorption_energies_context(self, node_ids: List[int]) -> str:
        """
        Retrieve adsorption energies context for the given node IDs.

        Args:
            node_ids (List[int]): The list of node IDs to get context for.

        Returns:
            str: A string containing adsorption energies context.
        """
        pass

    def get_reaction_pathways_context(self, node_ids: List[int]) -> str:
        """
        Retrieve reaction pathways context for the given node IDs.

        Args:
            node_ids (List[int]): The list of node IDs to get context for.

        Returns:
            str: A string containing reaction pathways context.
        """
        pass