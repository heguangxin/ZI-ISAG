# ZI-ISAG
 ZI-ISAG is an internet search augmented generation paradigm for LLMs. Key features include:
 - Dynamically integrate the latest online information (without maintaining any index for any fixed corpus).
 - The core component is an extractor LLM which can accurately and efficiently extract relevant information from the tagged content.
 - Extractor LLM only outputs the associated TAGs in the tagged content, significantly improving efficiency of the extraction process.

----------

This project was made possible thanks to a collaboration with:
<img src="https://github.com/Relaxed-System-Lab/ZI-ISAG/blob/main/images/collaboration.png" width="750" height="90" alt="01_hkust_fdu_tsinghua">

----------

## Content
- [Paradigm](#paradigm)
- [Extractor LLM](#extractor-llm)
- [Environment](#environment)
- [Instruction Set Construction](#instruction-set-construction)
- [Train](#train)

## Paradigm
<img src="https://github.com/Relaxed-System-Lab/ZI-ISAG/blob/main/images/Search-RAG-Internet.png" width="455" height="217" alt="zi-isag-paradigm">

## Extractor LLM
<img src="https://github.com/Relaxed-System-Lab/ZI-ISAG/blob/main/images/case.png" width="455" height="266" alt="extractor-llm-case">

## Enviroment
We use conda to manage our enviroment with CUDA 12.1 and cuDNN 9.1.0.

Create and activate env zi-isag:
```bash
conda create -p /YOUR_PATH_TO_CONDA_ENV/zi-isag python=3.10.14
conda activate /YOUR_PATH_TO_CONDA_ENV/zi-isag
```
Build the env with requirements.txt
```bash
python -m pip install -r requirements.txt
```

## Instruction Set Construction
The construction includes the following steps:
### Split and Tag
### Extract Info-Tag
### Double Check
### Portion and Construct
All details of these steps could be found in `data/construct.py` 

