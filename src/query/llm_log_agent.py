"""
This module contains the LLMLogAgent class for processing queries related to catalyst systems
and interacting with the NodeContext to retrieve relevant information.
"""

import re
from typing import List, Dict, Any
from node_context import NodeContext
from azure_openai_handler import AzureOpenAIHandler
from micro_structure import MicroStructureAgent

# Load the logging configuration
from logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/application.log",
        log_config_path="src/query/logging_config.ini",
    )

logger = LogManager.get_logger("llm_log_agent")


class LLMLogAgent:
    """
    A class for processing queries related to catalyst systems and retrieving relevant information.

    This class interacts with NodeContext and MicroStructureAgent instances to extract catalyst information,
    determine query scope and context, and generate responses using AzureOpenAIHandler.

    Attributes:
        node_context (NodeContext): An instance of NodeContext for data retrieval and processing.
        azure_handler (AzureOpenAIHandler): An instance of AzureOpenAIHandler for generating responses.
        micro_agent (MicroStructureAgent): An instance of MicroStructureAgent for processing microstructure queries.
        valid_elements (set): A set of valid transition metal elements.
        valid_elements_lower (set): A set of lowercase valid transition metal elements.
    """

    def __init__(
        self,
        node_context: NodeContext,
        azure_handler: AzureOpenAIHandler,
        micro_agent: MicroStructureAgent,
    ):
        """
        Initialize the LLMLogAgent with NodeContext, AzureOpenAIHandler, and MicroStructureAgent instances.

        Args:
            node_context (NodeContext): An instance of NodeContext for data retrieval and processing.
            azure_handler (AzureOpenAIHandler): An instance of AzureOpenAIHandler for generating responses.
            micro_agent (MicroStructureAgent): An instance of MicroStructureAgent for processing microstructure queries.
        """
        self.node_context = node_context
        self.azure_handler = azure_handler
        self.micro_agent = micro_agent
        self.valid_elements = set(
            [
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "La",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Ac",
                "Rf",
                "Db",
                "Sg",
                "Bh",
                "Hs",
                "Mt",
                "Ds",
                "Rg",
                "Cn",
            ]
        )
        self.valid_elements_lower = set(elem.lower() for elem in self.valid_elements)
        logger.info("LLMLogAgent initialized")

    def process_query(self, query: str, context: str = None) -> str:
        """
        Process the given query and return a response.

        This method determines the scope and context type of the query,
        extracts relevant catalyst systems, retrieves appropriate nodes and context,
        and generates a response. It now also handles microstructure-related queries.

        Args:
            query (str): The input query to process.

        Returns:
            str: The generated response based on the query and retrieved context.

        Raises:
            NotImplementedError: If the query is microstructure-related but not for Cu or
            Zn.
        """
        logger.debug(f"Processing query: {query}")

        # Check if the query is related to microstructure
        microstructure_keywords = [
            "composition",
            "miller index",
            "miller indices",
            "surface",
            "preferable surface",
            "bulk",
            "site",
            "sites",
            "active site",
            "active sites",
        ]
        if any(keyword in query.lower() for keyword in microstructure_keywords):
            logger.info("Query identified as microstructure-related")
            filter_options = self.extract_catalyst_systems(query)
            valid_catalysts = [cat for cat in filter_options if cat in ["Cu", "Zn"]]
            if valid_catalysts:
                logger.info(
                    (
                        "Processing microstructure query for "
                        f" {', '.join(valid_catalysts)}: {query}"
                    )
                )
                return self.micro_agent.answer_query(query, valid_catalysts)
            else:
                logger.warning(
                    f"Microstructure query not implemented for: {filter_options}"
                )
                logger.info("Generating results for Cu-Zn catalyst.")
                valid_catalysts = ["Cu", "Zn"]
                return self.micro_agent.answer_query(query, valid_catalysts)

        context_type = self.determine_context_type(query)
        if not context:
            scope_type = self.determine_scope(query)
            # If not microstructure-related, process as before
            filter_options = self.extract_catalyst_systems(query)

            node_ids = self.node_context.get_nodes(scope_type, filter_options)
            context = self.node_context.get_context(node_ids, context_type)

        response = self.generate_response(query, context, context_type)
        logger.info(f"Generated response for query: {query}")
        return response

    def determine_scope(self, query: str) -> str:
        """
        Determine the scope of the query.

        Args:
            query (str): The input query.

        Returns:
            str: Either "best_path" or "all_nodes" depending on the query content.
        """
        if "best" in query.lower() or "optimal" in query.lower():
            return "best_path"
        else:
            return "all_nodes"

    def determine_context_type(self, query: str) -> str:
        """
        Determine the context type based on the query content.

        Args:
            query (str): The input query.

        Returns:
            str: The determined context type ("catalyst_recommendation", "adsorption_energies", or "reaction_pathways").
        """
        if "catalyst" in query.lower() or "recommendation" in query.lower():
            return "catalyst_recommendation"
        elif "adsorption" in query.lower() or "energy" in query.lower():
            return "adsorption_energies"
        elif "reaction" in query.lower() or "pathway" in query.lower():
            return "reaction_pathways"
        else:
            return "catalyst_recommendation"  # Default to catalyst recommendation if unclear

    def extract_catalyst_systems(self, query: str) -> List[str]:
        """
        Extract catalyst systems and individual elements from the query.

        This method identifies catalyst systems, individual elements, and their oxides
        mentioned in the query, focusing on transition metals.

        Args:
            query (str): The input query to analyze.

        Returns:
            List[str]: A list of identified catalyst systems and elements.
        """
        catalysts = []

        # Pattern for catalyst systems (e.g., Cu-Zn, Pt-Ru, cu-zn)
        system_pattern = r"\b([A-Za-z][a-z]?-[A-Za-z][a-z]?)\b"
        catalysts.extend(re.findall(system_pattern, query, re.IGNORECASE))

        # Pattern for individual elements, including in compound words (e.g., Cu, Pt, cu-based)
        element_pattern = r"\b([A-Za-z][a-z]?)(?:\b|-)"
        potential_elements = re.findall(element_pattern, query)
        for elem in potential_elements:
            if elem.lower() in self.valid_elements_lower:
                catalysts.append(elem)

        # Pattern for element oxides (e.g., ZnO, CuO, zno)
        oxide_pattern = r"\b([A-Za-z][a-z]?O)\b"
        potential_oxides = re.findall(oxide_pattern, query, re.IGNORECASE)
        catalysts.extend(
            [
                oxide
                for oxide in potential_oxides
                if oxide[:-1].lower() in self.valid_elements_lower
            ]
        )

        # Pattern for complex oxides (e.g., Cu2O, Fe3O4, cu2o)
        complex_oxide_pattern = r"\b([A-Za-z][a-z]?[2-9]?O[2-9]?)\b"
        potential_complex_oxides = re.findall(
            complex_oxide_pattern, query, re.IGNORECASE
        )
        catalysts.extend(
            [
                oxide
                for oxide in potential_complex_oxides
                if oxide[0].lower() in self.valid_elements_lower
            ]
        )

        # Standardize the output
        catalysts = [self.standardize_catalyst(cat) for cat in catalysts]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(catalysts))

    def standardize_catalyst(self, catalyst: str) -> str:
        """
        Standardize the capitalization of catalyst elements and oxides.

        Args:
            catalyst (str): The catalyst string to standardize.

        Returns:
            str: The standardized catalyst string.
        """
        # Capitalize the first letter of each element
        parts = catalyst.split("-")
        standardized_parts = []
        for part in parts:
            # Handle oxides
            if "o" in part.lower():
                element = part.rstrip("Oo123456789")
                if element.lower() in self.valid_elements_lower:
                    standardized_parts.append(
                        element.capitalize() + "O" + "".join(filter(str.isdigit, part))
                    )
            elif part.lower() in self.valid_elements_lower:
                standardized_parts.append(part.capitalize())
        return "-".join(standardized_parts)

    def generate_response(self, query: str, context: str, context_type: str) -> str:
        """
        Generate a response based on the query, context, and context type using AzureOpenAIHandler.

        Args:
            query (str): The original input query.
            context (str): The context information retrieved from NodeContext.
            context_type (str): The type of context being used.

        Returns:
            str: The generated response from the Azure OpenAI service.
        """
        # Prepare the messages for the Azure OpenAI service
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant specializing in catalysis. "
                "Use the provided context to answer the user's query."
                "Only limit your response to the provided context."
                "If the context does not have any related information, "
                "respond by stating : 'The context does not provide any information about the query.'",
            },
            {
                "role": "user",
                "content": f"Context ({context_type}):\n{context}\n\nQuery: {query}",
            },
        ]

        # Generate the response using AzureOpenAIHandler
        response = self.azure_handler.generate_response(messages)
        return response


# Example usage
if __name__ == "__main__":
    json_file_path = "/anfhome/rounak.meyur/chemreasoner_results_processed/chemreasoner_results/search_tree_6.json"
    csv_file_path = "/anfhome/shared/chemreasoner/cu_zn_co_to_ethanol_from_scratch/reward_values.csv"
    env_path = ".env"

    node_context = NodeContext(json_file_path)
    azure_handler = AzureOpenAIHandler(env_path)
    micro_agent = MicroStructureAgent(csv_file_path, azure_handler)
    agent = LLMLogAgent(node_context, azure_handler, micro_agent)

    # Example queries
    queries = [
        "What is the average reward for Cu-Zn catalysts with (111) Miller index?",
        "Which catalyst composition has the highest reward for the (100) Miller index?",
        "Compare the average rewards of Cu and Zn catalysts across all Miller indices.",
        "What are the top 5 performing catalysts for the (110) Miller index?",
        "How does the reward distribution vary for different bulk compositions?",
        "What is the standard deviation of rewards for each unique surface type?",
        "What's the best catalyst recommendation?",
        "List the roles of Cu in optimal Cu-Zn catalyst system?",
        "How do Pt and Pd catalysts as optimal catalyst perform in this reaction?",
        "Compare the performance of Cu-Zn and Ni-Co as choices for optimal catalysts",
        "What's the effect of ZnO in the optimal catalyst?",
        "What types of promoters are used in cu-based optimal catalysts?",
        "Discuss the advantages of Ag-supported catalysts in optimal catalysts",
        "In this reaction, what catalysts work best?",
        "How does Au perform as an optimal catalyst?",
        "Tell me about optimal catalysts containing Pt-Ru alloys",
    ]

    for query in queries:
        response = agent.process_query(query)
        logger.info("--------------Query-Response--------------")
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}\n")
        logger.info("------------------------------------------")
