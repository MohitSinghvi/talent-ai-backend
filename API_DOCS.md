# API Reference

All endpoints are served from the FastAPI server at:

```
http://<host>:<port>/
```

---

## 1. Root

**GET** `/`

A simple welcome message.

<aside>
**Response**

```json
{
  "message": "Welcome to the Attrition Prediction API! Use /docs for API documentation."
}
```
</aside>

---

## 2. Risk Curve Data

**GET** `/risk_curve_data`

Returns the average attrition risk and headcount for each combination of age range and gender.

<aside>
**Response Schema**

| Field                    | Type    | Description                                             |
|--------------------------|---------|---------------------------------------------------------|
| `Age_Range`              | string  | Age bin label (e.g. `18-25`, `26-35`)                   |
| `OriginalGender`         | string  | Employee gender                                         |
| `Attrition_Risk`         | number  | Mean attrition probability (0–1)                        |
| `OriginalEmployeeNumber` | integer | Number of employees in the group                        |
| `Attrition_Percentage`   | number  | Mean attrition risk as a percentage (0–100)             |

```json
[
  {
    "Age_Range": "26-35",
    "OriginalGender": "Male",
    "Attrition_Risk": 0.12,
    "OriginalEmployeeNumber": 23,
    "Attrition_Percentage": 12.0
  },
  ...
]
```
</aside>

---

## 3. Top Employees

**GET** `/top_employees_data`

Lists the top 10 employees most likely to leave, including their names, risk percentage, main risk factor, and department.

<aside>
**Response Schema**

| Field                      | Type    | Description                          |
|----------------------------|---------|--------------------------------------|
| `OriginalEmployeeNumber`   | integer | Employee ID                          |
| `FirstName`                | string  | First name                           |
| `LastName`                 | string  | Last name                            |
| `Attrition_Risk_Percentage`| number  | Risk as %                            |
| `Top_Contributing_Factor`  | string  | Feature driving the highest risk     |
| `Department`               | string  | Department name                      |

```json
[
  {
    "OriginalEmployeeNumber": 123,
    "FirstName": "Jane",
    "LastName": "Doe",
    "Attrition_Risk_Percentage": 87.3,
    "Top_Contributing_Factor": "MonthlyIncome",
    "Department": "Sales"
  },
  ...
]
```
</aside>

---

## 4. Department Pie Data

**GET** `/department_pie_data`

Provides each department’s mean attrition risk, suitable for a pie chart.

<aside>
**Response Schema**

| Field      | Type    | Description             |
|------------|---------|-------------------------|
| `Department` | string | Department name        |
| `Percentage` | number | Mean attrition risk (%) |

```json
[
  { "Department": "Sales", "Percentage": 15.2 },
  { "Department": "Research & Development", "Percentage": 9.8 },
  ...
]
```
</aside>

---

## 5. Model Evaluation

**GET** `/model_evaluation`

Returns classification metrics for the trained XGBoost attrition model.

<aside>
**Response Schema**

| Field             | Type    | Description                              |
|-------------------|---------|------------------------------------------|
| `Accuracy`        | number  | Overall accuracy (0–1)                   |
| `Precision`       | number  | Positive predictive value                |
| `Recall`          | number  | True positive rate                       |
| `F1 Score`        | number  | Harmonic mean of precision/recall        |
| `ROC-AUC Score`   | number  | Area under ROC curve                     |
| `Confusion Matrix`| array   | 2×2 confusion matrix                     |

```json
{
  "Accuracy": 0.89,
  "Precision": 0.82,
  "Recall": 0.78,
  "F1 Score": 0.80,
  "ROC-AUC Score": 0.92,
  "Confusion Matrix": [[120, 15], [10, 30]]
}
```
</aside>

---

## 6. Predictive Hiring

**GET** `/predictive_hiring`

Suggests hires equal to the count of at-risk employees and returns dummy candidate data.

<aside>
**Response Schema**

| Field                  | Type    | Description                           |
|------------------------|---------|---------------------------------------|
| `number_of_hires`      | integer | Count of employees predicted to leave |
| `suggested_candidates` | array   | List of candidate objects             |

```json
{
  "number_of_hires": 5,
  "suggested_candidates": [
    { "CandidateID": 101, "Name": "Alice Smith", "Experience": "5 years", "Skill": "Sales" },
    ...
  ]
}
```
</aside>

---

## 7. Top Risk Factors

**GET** `/top_risk_factors?top_n=<int>`

Retrieves the most important features from the attrition model.

- **Query Parameters**
  - `top_n` (integer, default `5`)

<aside>
**Response Schema**

| Field         | Type   | Description                          |
|---------------|--------|--------------------------------------|
| `risk_factor` | string | Name of the feature                  |
| `importance`  | number | Importance score (normalized)        |

```json
[
  { "risk_factor": "MonthlyIncome", "importance": 0.2311 },
  { "risk_factor": "JobSatisfaction", "importance": 0.1897 },
  ...
]
```
</aside>

---

## 8. Talent Pool

**GET** `/talent_pool`

Provides paginated access to recruitment applicants, with search and sorting.

- **Query Parameters**
  - `page` (integer, ≥1; default `1`)
  - `limit` (integer, ≥1; default `6`)
  - `search` (string; default `""`)
  - `sort_by` (`matchScore` | `experience` | `education`; default `matchScore`)
  - `sort_order` (`asc` | `desc`; default `desc`)

<aside>
**Response Schema**

| Field        | Type    | Description                     |
|--------------|---------|---------------------------------|
| `candidates` | array   | List of applicant objects       |
| `total`      | integer | Total matching records          |
| `page`       | integer | Current page number             |
| `totalPages` | integer | Total number of pages           |

**Applicant Object**

| Field        | Type    | Description                             |
|--------------|---------|-----------------------------------------|
| `id`         | integer | Applicant ID                            |
| `name`       | string  | Full name                               |
| `role`       | string  | Job title                               |
| `experience` | string  | Years of experience (e.g. “4 years”)    |
| `education`  | string  | Education level                         |
| `location`   | string  | City, State                             |
| `skills`     | array   | List of skills                          |
| `matchScore` | integer | Custom match score (0–100)              |
| `availability` | string| Availability status                     |
| `image`      | string  | Placeholder image URL                   |

```json
{
  "candidates": [
    { "id": 1001, "name": "Scott Sheppard", "role": "Research Director", ... }
  ],
  "total": 42,
  "page": 1,
  "totalPages": 7
}
```
</aside>

---

## 9. Hire Alternatives

**GET** `/hire_alternatives`

For each of the top 5 high‑risk employees, returns up to 3 skill‑matched candidate alternatives.

<aside>
**Response Schema**

Array of objects:

| Field          | Type   | Description                            |
|----------------|--------|----------------------------------------|
| `employee`     | object | High‑risk employee details             |
| `alternatives` | array  | Top 3 matched candidate suggestions    |

**Employee Object**

| Field        | Type    | Description                          |
|--------------|---------|--------------------------------------|
| `name`       | string  | Employee full name                   |
| `role`       | string  | Job role                             |
| `department` | string  | Department name                      |
| `tenure`     | string  | Tenure at company (e.g. “3 years”)   |
| `riskScore`  | number  | Attrition risk percentage (0–100)    |
| `reason`     | string  | Primary risk factor                  |
| `skills`     | array   | Employee’s skill list                |
| `image`      | string  | Placeholder image URL                |

**Candidate Object**

| Field        | Type    | Description                           |
|--------------|---------|---------------------------------------|
| `name`       | string  | Candidate full name                   |
| `role`       | string  | Job title                             |
| `experience` | string  | Years of experience                   |
| `matchScore` | number  | Combined skill/experience match (%)   |
| `skills`     | array   | Candidate skill list                  |
| `image`      | string  | Placeholder image URL                 |

```json
[
  {
    "employee": { "name":"Edward Buck","role":"Laboratory Technician", ... },
    "alternatives": [
      { "name":"Sarah Wilson","role":"Senior Software Engineer", ... },
      ...
    ]
  },
  ...
]
```
</aside>

---

## 10. Department Overview

**GET** `/department_data`

Lists each department’s summary, including a preview of up to 5 employees, manager info, and key metrics.

<aside>
**Response Schema**

Array of department objects:

| Field        | Type    | Description                          |
|--------------|---------|--------------------------------------|
| `name`       | string  | Department name                      |
| `headCount`  | integer | Total employees                      |
| `manager`    | object  | Manager’s details                    |
| `employees`  | array   | Preview of up to 5 employee objects  |
| `metrics`    | object  | Department metrics                   |

**Manager Object**

| Field  | Type   | Description      |
|--------|--------|------------------|
| `name` | string | Manager name     |
| `email`| string | Manager email    |
| `phone`| string | Manager phone    |
| `image`| string | Manager picture  |

**Metrics Object**

| Field          | Type   | Description                               |
|----------------|--------|-------------------------------------------|
| `attritionRate`| string | Mean attrition rate (e.g. “8.5%”)         |
| `avgTenure`    | string | Average tenure (e.g. “3.2 years”)         |
| `openPositions`| integer| Estimated open positions                  |

**Employee Preview Object**

| Field    | Type   | Description           |
|----------|--------|-----------------------|
| `name`   | string | Full name             |
| `role`   | string | Job role              |
| `email`  | string | Generated email       |
| `location`| string| Placeholder location  |
| `image`  | string | Placeholder image     |

```json
[
  {
    "name": "Research & Development",
    "headCount": 50,
    "manager": { ... },
    "employees": [ ... ],
    "metrics": { ... }
  },
  ...
]
```
</aside>
