import json
import re
from typing import List, Dict, Any, Optional

import logging
console = logging.StreamHandler()
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)


class NodeContext:
    def __init__(self, json_file_path: str):
        self.json_data = self.load_json(json_file_path)
        self.specific_usage = self.extract_specific_usage()

    def load_json(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)
        
    def extract_specific_usage(self) -> str:
        template = self.json_data.get('template', '')
        match = re.search(r'Generate a list of top-5 .+ for (.+)\. {include_statement}', template)
        if match:
            return match.group(1)
        return "CO2 hydrogenation reaction to methanol" # default value if not found

    def get_nodes(self, scope_type: str, filter_options: Optional[List[str]] = None) -> List[int]:
        if scope_type == "best_path":
            node_ids = self.get_best_path_node_ids()
        else:
            node_ids = self.get_all_node_ids()
        if filter_options:
            return self.filter_nodes(node_ids, filter_options)
        return node_ids

    def filter_nodes(self, node_ids: List[int], filter_options: List[str]) -> List[int]:
        filtered_ids = []
        for node_id in node_ids:
            node = self.find_node_by_id(node_id)
            if node:
                answer = node['info']['generation'][0]['answer']
                if self.contains_exact_match(answer, filter_options):
                    filtered_ids.append(node_id)
        
        return filtered_ids if filtered_ids else node_ids  # Return all nodes if no matches found

    def contains_exact_match(self, text: str, options: List[str]) -> bool:
        for option in options:
            # Create a regex pattern that matches the option as a whole word
            pattern = r'\b' + re.escape(option) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def get_best_path_node_ids(self) -> List[int]:
        best_node = self.find_best_node(self.json_data)
        if best_node is None:
            return []
        return self.trace_path_to_root(best_node)

    def find_best_node(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        best_node = node
        if 'children' in node:
            for child in node['children']:
                child_best = self.find_best_node(child)
                if child_best['node_rewards'] > best_node['node_rewards']:
                    best_node = child_best
        return best_node

    def trace_path_to_root(self, node: Dict[str, Any]) -> List[int]:
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
        if 'children' in node:
            for child in node['children']:
                if child['id'] == target_id:
                    return node
                parent = self.find_parent(child, target_id)
                if parent:
                    return parent
        return None

    def get_all_node_ids(self) -> List[int]:
        all_node_ids = []
        queue = [self.json_data]
        while queue:
            node = queue.pop(0)
            all_node_ids.append(node['id'])
            queue.extend(node.get('children', []))
        return all_node_ids

    def get_context(self, nodes: List[Dict[str, Any]], context_type: str) -> List[Dict[str, Any]]:
        if context_type == "catalyst_recommendation":
            return self.get_catalyst_recommendation_context(nodes)
        elif context_type == "adsorption_energies":
            return self.get_adsorption_energies_context(nodes)
        elif context_type == "reaction_pathways":
            return self.get_reaction_pathways_context(nodes)
        else:
            raise ValueError(f"Unknown context type: {context_type}")

    def find_node_by_id(self, node_id: int) -> Dict[str, Any]:
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
        context_parts = ["Catalyst Recommendation Context:\n"]
        context_parts.append(f"This context provides information about catalyst recommendations for {self.specific_usage}. ")
        context_parts.append("It includes node IDs, corresponding answers, and node rewards for each step in the decision path.\n")

        for node_id in node_ids:
            node = self.find_node_by_id(node_id)
            if node:
                context_parts.append(f"Answer: {node['info']['generation'][0]['answer']}")
                context_parts.append(f"Node Reward: {node['node_rewards']}")
                context_parts.append("")  # Empty line for separation

        return "\n".join(context_parts)

    def get_adsorption_energies_context(self, nodes: List[Dict[str, Any]]) -> str:
        # Placeholder: extract adsorption energy data from the nodes
        pass

    def get_reaction_pathways_context(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Placeholder: extract reaction pathway data from the nodes
        pass



class LLMLogAgent:
    def __init__(self, node_context: NodeContext):
        self.node_context = node_context
        self.valid_elements = set([
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'
        ])
        self.valid_elements_lower = set(elem.lower() for elem in self.valid_elements)

    def process_query(self, query: str) -> str:
        scope_type = self.determine_scope(query)
        context_type = self.determine_context_type(query)
        filter_options = self.extract_catalyst_systems(query)

        logger.debug(f"Context type : {context_type}")
        logger.debug(f"Scope type : {scope_type}")
        logger.debug(f"Filters on catalyst : {filter_options}")
        
        node_ids = self.node_context.get_nodes(scope_type, filter_options)
        logger.debug(f"After filter: {len(node_ids)}")
        return
        context = self.node_context.get_context(node_ids, context_type)
        
        return self.generate_response(query, context, context_type)

    def determine_scope(self, query: str) -> str:
        if "best" in query.lower() or "optimal" in query.lower():
            return "best_path"
        else:
            return "all_nodes"

    def determine_context_type(self, query: str) -> str:
        if "catalyst" in query.lower() or "recommendation" in query.lower():
            return "catalyst_recommendation"
        elif "adsorption" in query.lower() or "energy" in query.lower():
            return "adsorption_energies"
        elif "reaction" in query.lower() or "pathway" in query.lower():
            return "reaction_pathways"
        else:
            return "catalyst_recommendation"  # Default to catalyst recommendation if unclear
    
    def extract_catalyst_systems(self, query: str) -> List[str]:
        catalysts = []
        
        # Pattern for catalyst systems (e.g., Cu-Zn, Pt-Ru, cu-zn)
        system_pattern = r'\b([A-Za-z][a-z]?-[A-Za-z][a-z]?)\b'
        catalysts.extend(re.findall(system_pattern, query, re.IGNORECASE))
        
        # Pattern for individual elements, including in compound words (e.g., Cu, Pt, cu-based)
        element_pattern = r'\b([A-Za-z][a-z]?)(?:\b|-)'
        potential_elements = re.findall(element_pattern, query)
        for elem in potential_elements:
            if elem.lower() in self.valid_elements_lower:
                catalysts.append(elem)
        
        # Pattern for element oxides (e.g., ZnO, CuO, zno)
        oxide_pattern = r'\b([A-Za-z][a-z]?O)\b'
        potential_oxides = re.findall(oxide_pattern, query, re.IGNORECASE)
        catalysts.extend([oxide for oxide in potential_oxides if oxide[:-1].lower() in self.valid_elements_lower])
        
        # Pattern for complex oxides (e.g., Cu2O, Fe3O4, cu2o)
        complex_oxide_pattern = r'\b([A-Za-z][a-z]?[2-9]?O[2-9]?)\b'
        potential_complex_oxides = re.findall(complex_oxide_pattern, query, re.IGNORECASE)
        catalysts.extend([oxide for oxide in potential_complex_oxides if oxide[0].lower() in self.valid_elements_lower])
        
        # Standardize the output
        catalysts = [self.standardize_catalyst(cat) for cat in catalysts]
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(catalysts))
    
    def standardize_catalyst(self, catalyst: str) -> str:
        # Capitalize the first letter of each element
        parts = catalyst.split('-')
        standardized_parts = []
        for part in parts:
            # Handle oxides
            if 'o' in part.lower():
                element = part.rstrip('Oo123456789')
                if element.lower() in self.valid_elements_lower:
                    standardized_parts.append(element.capitalize() + 'O' + ''.join(filter(str.isdigit, part)))
            elif part.lower() in self.valid_elements_lower:
                standardized_parts.append(part.capitalize())
        return '-'.join(standardized_parts)

    def generate_response(self, query: str, context: List[Dict[str, Any]], context_type: str) -> str:
        # Implement the LLM handler to generate a response using a prompt
        return f"Response based on {context_type} context: {context}"



if __name__ == "__main__":
    import sys
    path_to_JSON_file = f"/anfhome/rounak.meyur/chemreasoner_results_processed/chemreasoner_results/search_tree_{sys.argv[1]}.json"
    node_context = NodeContext(json_file_path=path_to_JSON_file)

    agent = LLMLogAgent(node_context)
    
    # Example queries
    queries = [
        "What's the best catalyst recommendation?",
        "List the roles of ZnO in Cu-Zn catalyst system?",
        "How do Pt and Pd catalysts perform in this reaction?",
        "Compare the performance of Cu-Zn and Ni-Co catalysts", 
        "What are the proposed active sites for Cu-Zn based catalysts?",
        "What can cause high activity in the design of Cu/zno catalysts?",
        "What types of promoters are used in cu-based catalysts?",
        "What are the evidences of SMSI in the catalyst?"
    ]
    
    for query in queries:
        logger.debug(f"Query: {query}")
        agent.process_query(query)
        # response = agent.process_query(query)