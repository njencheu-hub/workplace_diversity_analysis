# Workplace Diversity & Salary Prediction

This Python project analyzes a simulated company's workforce structure and uses machine learning to predict employee salaries while exploring fairness across roles, departments, and gender.

---

## Table of Contents

- [Objective](#-objective)
- [Real-World Use Cases](#-real-world-use-cases)
- [Datasets](#-datasets)
- [Features](#-features)
  - [1. Organizational Mapping](#1-organizational-mapping)
  - [2. Dataset Engineering](#2-dataset-engineering)
  - [3. Salary Prediction Model](#3-salary-prediction-model)
  - [4. Fairness & Diversity Insight](#4-fairness--diversity-insight)
- [Visuals](#-visuals)
- [Outputs](#-outputs)
- [Tools Used](#️-tools-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [License](#-license)

---

##  Objective
To uncover salary drivers, classify employees into hierarchical levels, assess management responsibility, and evaluate fairness across gender and experience levels using predictive modeling.

---

## Real-World Use Cases

- **HR Analytics**: Benchmark salaries based on role, experience, and department.
- **Fairness Audits**: Identify structural pay disparities across demographic groups.
- **Compensation Planning**: Support data-driven decisions for equitable salary adjustments.
- **Organizational Design**: Evaluate the management burden using direct/indirect reports data.
> Ideal for HR analytics, compensation planning, or fairness audits in corporate data science
---

##  Datasets
- `company_hierarchy.csv`: Organizational structure with employee-manager relationships and department info.
- `employee.csv`: Employee-level salary, education, gender, and experience details.

---

##  Features

### 1. Organizational Mapping
- Classified 10,000 employees into 6 hierarchical levels (IC to CEO).
- Calculated number of direct and indirect reports per employee.

### 2. Dataset Engineering
- Merged hierarchy and employee data.
- Encoded ordered categories: education level, company level, and gender.
- One-hot encoded department for model interpretability.

### 3. Salary Prediction Model
- Built a Random Forest Regressor to predict salaries.
- Evaluated with MSE, explained variance, and practical accuracy (within ±25% of actual salary).
- Identified most influential variables: department, experience, and reports managed.

### 4. Fairness & Diversity Insight
- Explored average salary gaps across gender and departments.
- Concluded that salary disparities stemmed from role distribution, not bias.
- Suggested HR improve gender balance across high-paying departments and revisit experience-based raises.

---

##  Visuals
- Feature importance plot
- Partial dependence plots for `department`, `experience`, and `gender`
- Gender-pay breakdown by department

##  Outputs
- `variable_importance.png`
- `partial_plot_for_dept.png`
- `partial_plot_for_yrs_of_experience.png`
- `partial_dependence_plot_for_sex.png`

## Tools Used

- **Python**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Machine Learning**: Random Forest Regression, `PartialDependenceDisplay`
- **Data Engineering**: Merging, encoding, feature creation

---

## Installation


To install the required Python packages:


pip install -r requirements.txt


## Usage


To run the analysis:


python staffing_Clipboard-Health.py


## Contributing
We welcome community contributions!


- Fork the repository


- Create a new branch:


git checkout -b feature/your-feature


- Make your changes


- Push to your branch:


git push origin feature/your-feature


- Submit a Pull Request


## License
This project is licensed under the MIT License.
