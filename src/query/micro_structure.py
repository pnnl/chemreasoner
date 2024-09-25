"""
This module contains the MicroStructureAgent class for processing queries related to
catalyst microstructures and mapping them to pandas queries.
"""

import pandas as pd
import ast
from typing import List, Dict, Any
from azure_openai_handler import AzureOpenAIHandler
from logging_utils import LogManager

# Load the logging configuration
if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_microstruct_agent.log", 
        log_config_path="src/query/logging_config.ini"
        )

logger = LogManager.get_logger("microstruct_agent")


class MicroStructureAgent:
    """
    A class for processing queries related to catalyst microstructures and mapping them to pandas queries.

    This class loads a CSV file as a pandas DataFrame and uses an AzureOpenAIHandler
    to generate and execute pandas queries based on natural language input.

    Attributes:
        df (pd.DataFrame): The pandas DataFrame containing the microstructure data.
        azure_handler (AzureOpenAIHandler): An instance of AzureOpenAIHandler for generating queries.
        relevant_columns (List[str]): A list of relevant column names in the DataFrame.
    """

    def __init__(self, csv_file_path: str, azure_handler: AzureOpenAIHandler):
        """
        Initialize the MicroStructureAgent with a CSV file and AzureOpenAIHandler.

        Args:
            csv_file_path (str): Path to the CSV file containing microstructure data.
            azure_handler (AzureOpenAIHandler): An instance of AzureOpenAIHandler for generating queries.
        """
        # Load the CSV file and filter out rows where 'reward' is NaN
        self.df = pd.read_csv(csv_file_path)
        original_row_count = len(self.df)
        self.df = self.df.dropna(subset=['reward'])
        filtered_row_count = len(self.df)
        rows_removed = original_row_count - filtered_row_count
        
        logger.info(f"Loaded CSV file from {csv_file_path}")
        logger.info(f"Removed {rows_removed} rows with NaN rewards")
        logger.info(f"Remaining rows: {filtered_row_count}")

        self.azure_handler = azure_handler
        self.relevant_columns = self.get_relevant_columns()
        self.preprocess_miller_indices()
        logger.info("MicroStructureAgent initialized with filtered CSV data")

    def get_relevant_columns(self) -> List[str]:
        """
        Get the list of relevant column names from the DataFrame.

        Returns:
            List[str]: A list of relevant column names.
        """
        all_columns = self.df.columns
        relevant_columns = [col for col in all_columns if col in [
            "millers", "surface", "site_placement", "bulk_composition",
            "bulk_symmetry", "site_composition", "reward"
        ]]
        return relevant_columns

    def preprocess_miller_indices(self):
        """
        Preprocess the Miller indices in the DataFrame.

        This method converts the Miller indices from string representation (e.g., '(1,1,1)')
        to a tuple of integers and adds a new column 'miller_index' with the combined representation.
        """
        if 'millers' in self.df.columns:
            self.df['millers'] = self.df['millers'].apply(ast.literal_eval)
            self.df['miller_index'] = self.df['millers'].apply(lambda x: f"({x[0]}{x[1]}{x[2]})")

    def answer_query(self, query: str, catalysts: List[str]) -> str:
        """
        Process a natural language query and return the result.

        This method uses the AzureOpenAIHandler to generate a pandas query based on the
        natural language input, executes the query, and returns the result.

        Args:
            query (str): The natural language query to process.
            catalysts (List[str]): List of catalyst names identified in the query.

        Returns:
            str: The result of the query execution or an error message.
        """
        miller_index = self.extract_miller_index(query)
        logger.info(f"Extracted Miller index: {miller_index}")

        prompt = f"""
        Given the following pandas DataFrame columns: {', '.join(self.relevant_columns + ['miller_index'])}
        
        The 'millers' column contains tuples like (1,1,1), and 'miller_index' contains strings like '(111)'.
        The 'bulk_composition' column contains the catalyst compositions as strings.
        The 'reward' column contains numerical values representing the performance of each catalyst configuration.

        Generate a pandas query to answer the following question: "{query}"
        
        Consider the following aspects when generating the query:
        1. If a specific Miller index ({miller_index}) is mentioned, filter for it.
        2. If specific catalysts ({', '.join(catalysts)}) are mentioned, filter rows where the 'bulk_composition' column contains any of these catalysts.
        3. The query might ask for:
           - Average rewards
           - Maximum or minimum rewards
           - Specific reward values for certain conditions
           - Comparisons between different catalysts or conditions
           - Top N performing catalysts
           - Distribution of rewards (e.g., quartiles, standard deviation)
        4. The query might require grouping by one or more columns (e.g., 'bulk_composition', 'miller_index', 'surface', etc.)
        5. The query might ask for sorting the results in a specific order.

        Ensure your pandas query addresses all relevant aspects of the question. Include filtering, grouping, aggregation, and sorting operations as necessary.

        Return only the Python code for the pandas query, without any explanations.
        Make sure to use 'df' as the DataFrame name in your query.
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates pandas queries."},
            {"role": "user", "content": prompt}
        ]

        try:
            pandas_query = self.azure_handler.generate_response(messages)
            logger.info(f"Generated pandas query: {pandas_query}")

            df = self.df  # Create a local variable 'df' that refers to self.df
            result = eval(pandas_query)
            
            if isinstance(result, pd.DataFrame):
                # Filter the result to include only relevant columns
                result = result[[col for col in self.relevant_columns if col in result.columns]]
                return result.to_string()
            elif isinstance(result, pd.Series):
                return result.to_string()
            else:
                return str(result)
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def extract_miller_index(self, query: str) -> str:
        """
        Extract the Miller index from the query string.

        Args:
            query (str): The input query string.

        Returns:
            str: The extracted Miller index, or None if not found.
        """
        import re
        match = re.search(r'\((\d{3})\)', query)
        if match:
            return match.group(1)
        return None

# Example usage
if __name__ == "__main__":
    csv_file_path = "/anfhome/shared/chemreasoner/cu_zn_co2_to_methanol_from_scratch/reward_values.csv"
    env_path = ".env"  # Replace with the path to your .env file

    azure_handler = AzureOpenAIHandler(env_path)
    micro_agent = MicroStructureAgent(csv_file_path, azure_handler)

    # Example queries
    queries = [
        "What is the average reward for Cu-Zn catalysts with (111) Miller index?",
        "Which catalyst composition has the highest reward for the (100) Miller index?",
        "Compare the average rewards of Cu and Zn catalysts across all Miller indices.",
        "What are the top 5 performing catalysts for the (110) Miller index?",
        "How does the reward distribution vary for different bulk compositions?",
        "What is the standard deviation of rewards for each unique surface type?"
    ]

    for query in queries:
        logger.info(f"Query: {query}")
        catalysts = ["Cu", "Zn"]  # This would typically come from LLMLogAgent
        result = micro_agent.answer_query(query, catalysts)
        logger.info(f"Result:\n{result}\n")