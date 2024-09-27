"""
This module contains the MicroStructureAgent class for processing queries related to
catalyst microstructures and mapping them to pandas queries.
"""

import pandas as pd
import numpy as np
import ast
import re
from typing import List, Dict, Any, Tuple
from azure_openai_handler import AzureOpenAIHandler

# Load the logging configuration
from logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_microstructure.log", 
        log_config_path="src/query/logging_config.ini"
        )

logger = LogManager.get_logger("microstructure")

class MicroStructureAgent:
    """
    A class for processing queries related to catalyst microstructures and mapping them to pandas queries.

    This class loads a CSV file as a pandas DataFrame and uses an AzureOpenAIHandler
    to generate and execute pandas queries based on natural language input.

    Attributes:
        df (pd.DataFrame): The pandas DataFrame containing the microstructure data.
        azure_handler (AzureOpenAIHandler): An instance of AzureOpenAIHandler for generating queries.
        relevant_columns (List[str]): A list of relevant column names in the DataFrame.
        adsorbate_symbols (List[str]): A list of acceptable adsorbate symbols.
        column_of_interest (str): The column currently being analyzed (either 'reward' or an adsorbate energy column).
    """

    def __init__(self, csv_file_path: str, azure_handler: AzureOpenAIHandler):
        """
        Initialize the MicroStructureAgent with a CSV file and AzureOpenAIHandler.

        Args:
            csv_file_path (str): Path to the CSV file containing microstructure data.
            azure_handler (AzureOpenAIHandler): An instance of AzureOpenAIHandler for generating queries.
        """
        # Load the CSV file with specific columns
        self.df = pd.read_csv(csv_file_path)
        self.df["reward"] = np.log10(self.df["reward"])
        
        # Rename energy columns by removing the '*' symbol
        energy_columns = [col for col in self.df.columns if col.startswith('energy_*')]
        
        # Keep only the relevant columns
        relevant_columns = [
            "millers", "bulk_composition", "bulk_symmetry", "site_composition", "reward"
        ] + energy_columns
        self.df = self.df[relevant_columns]

        self.azure_handler = azure_handler
        self.relevant_columns = self.get_relevant_columns()
        self.adsorbate_symbols = self.get_adsorbate_symbols()
        self.column_of_interest = "reward"
        self.preprocess_miller_indices()
        logger.info("MicroStructureAgent initialized with filtered CSV data")

    def get_relevant_columns(self) -> List[str]:
        """
        Get the list of relevant column names from the DataFrame.

        Returns:
            List[str]: A list of relevant column names.
        """
        return ["millers", "bulk_composition", "bulk_symmetry", "site_composition", "reward", "miller_index"]

    def get_adsorbate_symbols(self) -> List[str]:
        """
        Get the list of acceptable adsorbate symbols from the DataFrame columns.

        Returns:
            List[str]: A list of acceptable adsorbate symbols.
        """
        return [col.split("*")[1] for col in self.df.columns if col.startswith("energy_*")]

    def filter_nan_rows(self):
        """
        Filter out rows where the column of interest has NaN values.
        """
        original_row_count = len(self.df)
        self.df = self.df.dropna(subset=[self.column_of_interest])
        filtered_row_count = len(self.df)
        rows_removed = original_row_count - filtered_row_count
        logger.debug(f"Removed {rows_removed} rows with NaN values in {self.column_of_interest}")
        logger.debug(f"Remaining rows: {filtered_row_count}")

    def preprocess_miller_indices(self):
        """
        Preprocess the Miller indices in the DataFrame.

        This method converts the Miller indices from string representation (e.g., '(1,1,1)')
        to a tuple of integers and adds a new column 'miller_index' with the combined representation.
        """
        if 'millers' in self.df.columns:
            self.df['millers'] = self.df['millers'].apply(ast.literal_eval)
            self.df['miller_index'] = self.df['millers'].apply(lambda x: f"({x[0]}{x[1]}{x[2]})")
        logger.debug("Preprocessed Miller indices")

    def answer_query(self, query: str, catalysts: List[str], max_attempts: int = 3) -> str:
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
        adsorbate = self.extract_adsorbate(query)
        is_active_sites_query, is_summary = self.is_active_sites_query(query)
        is_preferable_surface_query = "preferable surface" in query.lower()

        # Set the column of interest based on the adsorbate
        self.set_column_of_interest(adsorbate)

        logger.debug(f"Query: {query}")
        logger.debug(f"Extracted Miller index: {miller_index}")
        logger.debug(f"Extracted adsorbate: {adsorbate}")
        logger.debug(f"Is active sites query: {is_active_sites_query}")
        logger.debug(f"Is summary query: {is_summary}")
        logger.debug(f"Preferable surface query: {is_preferable_surface_query}")

        prompt = self.generate_prompt(query, catalysts, miller_index, is_active_sites_query, is_summary, is_preferable_surface_query)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates pandas queries."},
            {"role": "user", "content": prompt}
        ]

        index_attempts = 0
        while (index_attempts < max_attempts):
            try:
                pandas_query = self.azure_handler.generate_response(messages)
                logger.info(f"Attempt {index_attempts+1}: Generated pandas query: \n{pandas_query}")

                df = self.df.copy()  # Create a local variable 'df' that refers to self.df
                result = eval(pandas_query)
                if isinstance(result, str):
                    result = eval(result)
                
                response = self.format_response(result, is_active_sites_query, is_summary, is_preferable_surface_query, adsorbate, catalysts)
                return response
            
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                logger.error(error_msg)
                index_attempts += 1
                if index_attempts == max_attempts:
                    return error_msg

    def set_column_of_interest(self, adsorbate: str):
        """
        Set the column of interest based on the adsorbate.

        Args:
            adsorbate (str): The adsorbate symbol extracted from the query.
        """
        if adsorbate and f"energy_*{adsorbate}" in self.df.columns:
            self.column_of_interest = f"energy_*{adsorbate}"
        else:
            self.column_of_interest = "reward"
        self.filter_nan_rows()

    def generate_prompt(self, query: str, catalysts: List[str], miller_index: str, is_active_sites_query: bool, is_summary: bool, is_preferable_surface_query: bool) -> str:
        """
        Generate a prompt for the LLM based on the query and extracted information.

        Args:
            query (str): The original query.
            catalysts (List[str]): List of catalyst names.
            miller_index (str): Extracted Miller index.
            is_active_sites_query (bool): Whether the query is about active sites.
            is_summary (bool): Whether the query is asking for a summary.

        Returns:
            str: The generated prompt.
        """

        prompt = f"""
        Given the following pandas DataFrame columns: {', '.join(self.relevant_columns)}
        
        The 'millers' column contains tuples like (1,1,1), and 'miller_index' contains strings like '(111)'.
        The 'bulk_composition' column contains the catalyst compositions as strings.
        The column of interest is '{self.column_of_interest}' which contains numerical values.

        Generate a pandas query to answer the following question: "{query}"
        
        Consider the following aspects when generating the query:
        1. If a specific Miller index ({miller_index}) is mentioned, filter for it, otherwise ignore it.
        2. If specific catalysts ({', '.join(catalysts)}) are mentioned, filter rows where the 'bulk_composition' column contains any of these catalysts.
        3. Use the '{self.column_of_interest}' column for calculations and comparisons.
        4. When using groupby operations:
           - If you need to apply a function to the grouped data, use agg() instead of apply() where possible.
           - If you must use apply(), make sure to exclude the grouping columns from the operation.
           - Example of correct groupby usage:
             df.groupby('bulk_composition').agg({{'{self.column_of_interest}': ['mean', 'max', 'min']}})
           - If you need to use apply(), do it like this:
             df.groupby('bulk_composition')['{self.column_of_interest}'].apply(lambda x: x.nlargest(5))
        """

        if is_preferable_surface_query:
            prompt += f"""
            5. To find the preferable surface:
               - Group the data by 'bulk_composition'.
               - For each bulk composition, find the top 5 site compositions based on the '{self.column_of_interest}'.
               - Include the 'millers' column in the results to show the surface for each site.
               - Sort the results within each group by the '{self.column_of_interest}' in descending order.
            Example:
            df.groupby('bulk_composition').apply(lambda x: x.nlargest(5, '{self.column_of_interest}')[['site_composition', 'millers', '{self.column_of_interest}']]).reset_index(level=1, drop=True).sort_values('{self.column_of_interest}', ascending=False)
            """

        elif is_active_sites_query:
            if is_summary:
                prompt += f"""
                5. Group the data by 'bulk_composition'.
                6. For each group, find the top 5 values in the '{self.column_of_interest}' column.
                7. Display the corresponding 'site_composition' and 'millers' for these top 5 values.
                8. Sort the results by the '{self.column_of_interest}' column in descending order.
                Example:
                df.groupby('bulk_composition').apply(lambda x: x.nlargest(5, '{self.column_of_interest}')[['site_composition', 'millers', '{self.column_of_interest}']]).reset_index(level=1, drop=True).sort_values('{self.column_of_interest}', ascending=False)
                """
            else:
                prompt += f"""
                5. Find the top 5 values in the '{self.column_of_interest}' column.
                6. Display the rows with 'millers', 'bulk_composition', and 'site_composition' for these top 5 values.
                7. Sort the results by the '{self.column_of_interest}' column in descending order.
                Example:
                df.nlargest(5, '{self.column_of_interest}')[['millers', 'bulk_composition', 'site_composition', '{self.column_of_interest}']].sort_values('{self.column_of_interest}', ascending=False)
                """
        else:
            prompt += """
            5. The query might ask for:
               - Average values
               - Maximum or minimum values
               - Specific values for certain conditions
               - Comparisons between different catalysts or conditions
               - Distribution of values (e.g., quartiles, standard deviation)
            6. The query might require grouping by one or more columns (e.g., 'bulk_composition', 'miller_index', etc.)
            7. The query might ask for sorting the results in a specific order.
            """

        prompt += """
        Ensure your pandas query addresses all relevant aspects of the question. Include filtering, grouping, aggregation, and sorting operations as necessary.
        Make sure to use the correct groupby operations to avoid warnings about operating on grouping columns.

        Return only the python pandas query in a single line, without any indents and explanations.
        Make sure to use 'df' as the DataFrame name in your query.
        """

        return prompt

    def format_response(self, result: Any, is_active_sites_query: bool, is_summary: bool, is_preferable_surface_query: bool, adsorbate: str, catalysts: List[str]) -> str:
        """
        Format the response based on the query type and result.

        Args:
            result (Any): The result of the pandas query execution.
            is_active_sites_query (bool): Whether the query is about active sites.
            is_summary (bool): Whether the query is asking for a summary.
            adsorbate (str): The adsorbate symbol extracted from the query.
            query (str): The original query string.
            catalysts (List[str]): List of catalyst names identified in the query.

        Returns:
            str: The formatted response.
        """
        response = ""

        if adsorbate and f"energy_*{adsorbate}" not in self.df.columns:
            response += f"No information exists about the queried adsorbate ({adsorbate}). Using overall reward values instead.\n\n"

        if is_active_sites_query:
            response += "The top choices of active sites are:\n\n"
        elif is_preferable_surface_query:
            response += "The preferable surfaces for each bulk composition are:\n\n"

        if isinstance(result, pd.DataFrame):
            response += result.to_string() + "\n\n"
            response += self.summarize_dataframe(result, is_active_sites_query, is_summary, adsorbate, is_preferable_surface_query, catalysts)
        elif isinstance(result, pd.Series):
            response += result.to_string() + "\n\n"
            response += self.summarize_series(result, is_active_sites_query, adsorbate, is_preferable_surface_query, catalysts)
        else:
            response += str(result) + "\n\n"
            response += "The result is a single value or a non-tabular output.\n"

        return response

    def summarize_dataframe(self, df: pd.DataFrame, is_active_sites_query: bool, is_summary: bool, adsorbate: str, is_preferable_surface_query: bool, catalysts: List[str]) -> str:
        """
        Summarize the contents of a DataFrame result.

        Args:
            df (pd.DataFrame): The DataFrame to summarize.
            is_active_sites_query (bool): Whether the query is about active sites.
            is_summary (bool): Whether the query is asking for a summary.
            adsorbate (str): The adsorbate symbol extracted from the query.
            is_preferable_surface_query (bool): Whether the query is about preferable surfaces.
            catalysts (List[str]): List of catalyst names identified in the query.

        Returns:
            str: A summary of the DataFrame contents.
        """
        summary = "Summary of results:\n"

        if is_preferable_surface_query:
            summary += "- The results show the top 5 site compositions for each bulk composition.\n"
            
            if 'bulk_composition' in df.index.names:
                unique_compositions = df.index.get_level_values('bulk_composition').nunique()
                summary += f"- {unique_compositions} different bulk compositions are represented.\n"
            
            if 'site_composition' in df.columns:
                unique_sites = df['site_composition'].nunique()
                summary += f"- {unique_sites} unique site compositions are shown across all bulk compositions.\n"
            
            if 'millers' in df.columns:
                most_common_miller = df['millers'].mode().iloc[0]
                summary += f"- The most common Miller index across all compositions is {most_common_miller}.\n"
            
            # Determine which surface is preferable relative to the given catalysts
            if 'site_composition' in df.columns and catalysts:
                df['catalyst_match'] = df['site_composition'].apply(lambda x: any(cat in x for cat in catalysts))
                preferable_surface = "near" if df['catalyst_match'].any() else "far from"
                summary += f"- The preferable surface appears to be {preferable_surface} the catalysts in the list ({', '.join(catalysts)}).\n"

        elif is_active_sites_query:
            if is_summary:
                summary += f"- The results show a summary of active sites grouped by bulk composition.\n"
            else:
                summary += f"- The results show the top active sites across all compositions.\n"
            
            if 'bulk_composition' in df.columns:
                unique_compositions = df['bulk_composition'].nunique()
                summary += f"- {unique_compositions} different bulk compositions are represented.\n"
            
            if 'site_composition' in df.columns:
                unique_sites = df['site_composition'].nunique()
                summary += f"- {unique_sites} unique site compositions are shown.\n"

        if 'reward' in df.columns:
            avg_reward = df['reward'].mean()
            max_reward = df['reward'].max()
            min_reward = df['reward'].min()
            summary += f"- The average reward is {avg_reward:.2f}, with a maximum of {max_reward:.2f} and a minimum of {min_reward:.2f}.\n"

        if adsorbate and f'energy_*{adsorbate}' in df.columns:
            avg_energy = df[f'energy_*{adsorbate}'].mean()
            summary += f"- The average adsorption energy for {adsorbate} is {avg_energy:.2f}.\n"

        return summary

    def summarize_series(self, series: pd.Series, is_active_sites_query: bool, adsorbate: str) -> str:
        """
        Summarize the contents of a Series result.

        Args:
            series (pd.Series): The Series to summarize.
            is_active_sites_query (bool): Whether the query is about active sites.
            adsorbate (str): The adsorbate symbol extracted from the query.

        Returns:
            str: A summary of the Series contents.
        """
        summary = "Summary of results:\n"

        if is_active_sites_query:
            summary += "- The results show a single aspect of active sites.\n"

        if series.name == 'reward':
            avg_value = series.mean()
            max_value = series.max()
            min_value = series.min()
            summary += f"- The average reward is {avg_value:.2f}, with a maximum of {max_value:.2f} and a minimum of {min_value:.2f}.\n"
        elif adsorbate and series.name == f'energy_*{adsorbate}':
            avg_value = series.mean()
            summary += f"- The average adsorption energy for {adsorbate} is {avg_value:.2f}.\n"
        else:
            summary += f"- The series shows values for '{series.name}'.\n"
            summary += f"- It contains {len(series)} entries.\n"

        return summary

    def extract_miller_index(self, query: str) -> str:
        """
        Extract the Miller index from the query string.

        Args:
            query (str): The input query string.

        Returns:
            str: The extracted Miller index, or None if not found.
        """
        match = re.search(r'\((\d{3})\)', query)
        if match:
            return match.group(1)
        return None

    def extract_adsorbate(self, query: str) -> str:
        """
        Extract the adsorbate symbol from the query string.

        Args:
            query (str): The input query string.

        Returns:
            str: The extracted adsorbate symbol, or None if not found.
        """
        for adsorbate in self.adsorbate_symbols:
            if adsorbate in query:
                return adsorbate
        return None

    def is_active_sites_query(self, query: str) -> Tuple[bool, bool]:
        """
        Determine if the query is about active sites and if it's asking for a summary.

        Args:
            query (str): The input query string.

        Returns:
            Tuple[bool, bool]: (is_active_sites_query, is_summary)
        """
        is_active_sites = "active site" in query.lower() or "active sites" in query.lower()
        is_summary = "summary" in query.lower()
        return is_active_sites, is_summary
    
    

# Example usage
if __name__ == "__main__":
    csv_file_path = "/anfhome/shared/chemreasoner/cu_zn_co_to_methanol/reward_values.csv"  # Replace with your actual CSV file path
    env_path = ".env"  # Replace with the path to your .env file

    from azure_openai_handler import AzureOpenAIHandler
    azure_handler = AzureOpenAIHandler(env_path)
    micro_agent = MicroStructureAgent(csv_file_path, azure_handler)

    # Example queries
    queries = [
        "What are proposed active sites for Cu-Zn based catalysts?",
        "What are the active sites of CO in Cu-Zn catalysts?",
        "Give me a summary of active sites for H adsorption.",
        "Give a summary of preferable surface for CO adsorption",
        "Which is preferable surface - Cu rich or Zn rich?",
        # "Compare the average rewards of Cu and Zn catalysts across all Miller indices.",
        # "What are the top 5 performing catalysts for the (110) Miller index?",
        # "What is the average reward for Cu-Zn catalysts with (111) Miller index?",
    ]

    for query in queries:
        catalysts = ["Cu", "Zn"]  # This would typically come from LLMLogAgent
        result = micro_agent.answer_query(query, catalysts, max_attempts=5)
        logger.info(f"Query: {query}")
        logger.info(f"Result:\n{result}\n")