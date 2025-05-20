# ChemReasoner: Discovering catalysts via Generative AI and Computational Chemistry
![image](https://github.com/pnnl/chemreasoner/assets/7649924/ccae35c9-876e-4865-8e46-0b229167d522)

## News/Presentations/Publications
* NVIDIA GTC 2025 [Designing Transparent AI Systems for Scientific Discovery: Patterns and Practices for Human-AI Collaboration](https://www.nvidia.com/en-us/on-demand/session/gtc25-S72049)
* AAAI 2025: S. Beus, H. W. Sprueill, R. Meyur, M. V. Olarte, K. Agarwal, D. Zhang, U. Sanyal, J. Lercher, S. Choudhury, "Human-AI Interaction using Linguistic Reasoning and Computational Chemistry for Trustworthy Materials Discovery"
* PNNL Article: [Accelerating Materials Discovery with AI](https://www.pnnl.gov/publications/accelerating-materials-discovery-ai)
* PNNL Article: [Novel Computing Tool Learns the Language of Chemistry](https://www.pnnl.gov/publications/novel-computing-tool-learns-language-chemistry)
* ICML 2024: "Heuristic Search over a Large Language Model's Knowledge Space using Quantum-Chemical Feedback" [arXiv](https://arxiv.org/abs/2402.10980)
* Presentation at [MLCommons Science Working Group](https://sutanay.github.io/publications/ChemReasoner-SciMLCommons.pdf)
* We will have two presentations at upcoming American Chemical Society Spring 2024 National Meeting!
    * Sprueill H.W., C. Edwards, M.V. Olarte, U. Sanyal, H. Ji, and S. Choudhury. "Integrating generative AI with computational chemistry for catalyst design in biofuel/bioproduct applications." American Chemical Society Spring 2024 National Meeting, New Orleans, Louisiana (oral presentation).
    * Sprueill H.W., C. Edwards, M.V. Olarte, U. Sanyal, K. Agarwal, H. Ji, and S. Choudhury. 03/18/2024. "Extreme-Scale Heterogeneous Inference with Large Language Models and Atomistic Graph Neural Networks for Catalyst Discovery." American Chemical Society Spring 2024 National Meeting, New Orleans, Louisiana (poster). 
* Our work on Monte Carlo Thought Search is accepted for publication in EMNLP 2023 Findings ([arXiv](https://arxiv.org/abs/2310.14420))
* Excited to present "ChemReasoner: Large Language Model-driven Search over Chemical Spaces with Quantum Chemistry-guided Feedback" at [2023 Stanford Graph Learning Workshop](https://snap.stanford.edu/graphlearning-workshop-2023/)
* We are thrilled to be selected for the [Microsoft Accelerate Foundation Models Research Initiative](https://www.microsoft.com/en-us/research/collaboration/accelerating-foundation-models-research/)
* [Presentation](https://www.kisacoresearch.com/sites/default/files/presentations/aihwsummit-sutanay.pdf) at AI Hardware and Edge AI Summit, Santa Clara, September 2023

Citation
------

Please cite the following papers [https://arxiv.org/abs/2310.14420] [https://arxiv.org/abs/2402.10980] if you find our work useful.

```bibtex
@inproceedings{sprueill2023MCR,
  title={Monte Carlo Thought Search: Large Language Model Querying for Complex Scientific Reasoning in Catalyst Design},
  author={Sprueill, Henry W. and Edwards, Carl and Sanyal, Udishnu and Olarte, Mariefel and Ji, Heng and Choudhury, Sutanay}
  booktitle={In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP2023) Findings},
  year={2023}
}
@article{sprueill2024chemreasoner,
  title={CHEMREASONER: Heuristic Search over a Large Language Model's Knowledge Space using Quantum-Chemical Feedback},
  author={Sprueill, Henry W and Edwards, Carl and Agarwal, Khushbu and Olarte, Mariefel V and Sanyal, Udishnu and Johnston, Conrad and Liu, Hongbin and Ji, Heng and Choudhury, Sutanay},
  journal={arXiv preprint arXiv:2402.10980},
  year={2024}
}
```

## Installation Instructions

Installation assumes cuda version 12.0

```
mamba env create -f chemreasoner.yml
conda activate chemreasoner
git clone https://github.com/pnnl/chemreasoner.git
cd chemreasoner
git submodule update --init --recursive
cd ext/ocp/
pip install -e .
cd ../Open-Catalyst-Dataset
pip install -e .
cd ../..
```

To test the installation:
```
python src/scripts/test_gnn.py      # use --cpu to test on cpu only
```

## Running the ICML Code

The code to reproduce the ICML results is located in ```src/scripts/run_icml_queries.py```. An example run script has been provided in ```src/launch_scripts/run_icml.sh```. You will need to set a few parameters...

* savedir: The directory to save the results in
* start-query: The index of the first query to evaluate (see data/input_data/dataset.csv)
* start-query: The index of the final query to evaluate
* gnn-traj-dir: The directroy in which to store relaxation trajectories
* dotenv-path: The path to .env file containing api keys for your azure openai setup (see instructions below)

### Setting up .env file

The .env file should be located in the chemreasoner root directory and contain the api keys and info for your Azure OpenAI interface, which can be found on the Azure portal.

```
AZURE_OPENAI_DEPLOYMENT_NAME=<deployment name>
AZURE_OPENAI_ENDPOINT=<url to deployment endpoint>
AZURE_OPENAI_API_KEY=<api key>
AZURE_OPENAI_API_VERSION="2023-07-01-preview"
```

### Setting up local GNN server

To run relaxations with the GNN model, you will have to set up a redis server. To do so open a new terminal on the same machine that you will be running chemreasoner on (with access to a GPU). Then run,

```
redis-server --dir <directory to store redis server cache>
```

Here, ```--dir``` can be set to any directory.

### Running ICML code

Once you have set up the run script, the .env file, and started the local redis server, run the ICML code by entering

```
./src/launch_scripts/run_icml.sh
```



### Contact

[Sutanay Choudhury](https://www.linkedin.com/in/sutanay/)
sutanay tod choudhury ta pnnl tod gov
