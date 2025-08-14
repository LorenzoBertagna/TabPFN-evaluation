## Table of Contents

1. [Overview](#overview)
2. [Usage](#usage)

---

## Overview
This repository tries to evaluate the tabPFN imputation techniques. It will be compared with other imputation methods like XGBoost, Catboost, MICE and so on. The goal is to show which of these methods performs better, depending on different conditions, like the nan percantage, or the train size. 
All the code has been written in python.
To see all the work done, please refer to notion page of repository: ```https://www.notion.so/schneiderlab/Data-Imputation-Benchmark-1afa4699a977808486a1ecdc880a0fb6```

NOTE: Repository is in development.
---

 
## Usage
**Prerequisites**:
The code runs on the RWTH Cluster on the node login23-g-1.hpc.itc.rwth-aachen.de. This node has a 4 Invidia H100 Gpu that are used to run both TabPFN and Catboost. Thus, to run the scripts you need to have access to the RWTH Cluster and install the cuda package in python. Other packages like scikitlearn, tabPFN, pandas, numpy, catboost, xgboost, math are needed as well.

**Structure**
In the python script in src/MICE.py you find the main code used to assess the different imputation methods. 
Under src/tabpfn-extensions you find the new extensions created by the developer of tabPFN and by the whole community. Very important for us are the shap functionalities. Source of repository: ```https://github.com/PriorLabs/TabPFN```

**UML Class diagram**
![image info](src/UML_CD.png)
Used program: Plantuml


