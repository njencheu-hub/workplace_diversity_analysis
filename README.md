# Workplace Diversity & Salary Prediction

This Python project analyzes a simulated company's workforce structure and uses machine learning to predict employee salaries while exploring fairness across roles, departments, and gender.

##  Objective
To uncover salary drivers, classify employees into hierarchical levels, assess management responsibility, and evaluate fairness across gender and experience levels using predictive modeling.

##  Datasets
- `company_hierarchy.csv`: Organizational structure with employee-manager relationships and department info.
- `employee.csv`: Employee-level salary, education, gender, and experience details.

##  Key Steps

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
- Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn)
- Random Forest for modeling, PartialDependenceDisplay for interpretation

---

> Ideal for HR analytics, compensation planning, or fairness audits in corporate data science