# Predicting the Next Big Game
> [!NOTE]
> The original source code is not publicly available due to course/project restrictions.

Group Members: Omar Spiller Hernandez, Scott Lam, Evan Carey, David Gil

<table>
  <tr>
    <td rowspan="2">
      <img width="400" src="https://github.com/user-attachments/assets/828c5bae-441c-4a1f-8230-409e8056c537" />
    </td>
    <td>
      <img width="280" src="https://github.com/user-attachments/assets/7f3a0856-ba1e-4e31-b145-70f78a48ee87" />
    </td>
  </tr>
  <tr>
    <td>
      <img width="280" src="https://github.com/user-attachments/assets/828ac2df-9764-49d0-ae1c-ea040ad8be9f" />
    </td>
  </tr>
</table>

## Project Overview
The online game platform known as STEAM hosts over 97,000 games, making it a 
highly competitive environment for developers. Success on STEAM depends on 
factors like user reviews, critic scores and gameplay features. 
The goal here was to identigy which features correlate with a games success 
(measured by revenue and ratings). With this we built a predictive model 
to estimate a games potential for success.

We asked ourselves: Which features of a game are most predictive of a game's 
revenue and user ratings on Steam?

We predicted: Games with high Metacritic scores, positive user reviews and 
specific genre tags will correlate stronger with higher revenue and success.

## Selection of Data
### Dataset Source
- [Link to dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

    Source: Steam Games Dataset (Kaggle, retrieved via Steam API and Steam Spy).
    Size: 97,000+ games with attributes like price, reviews, Metacritic scores, and tags.

### Data Preprocessing

#### Cleaning
        Removed columns with non-analytical data (URLs, screenshots).
        Filtered out games with $0 price or missing critical data (Metacritic score = 0).
        Dropped extreme revenue outliers ("ELDEN RING", "Cyberpunk 2077").

#### Feature Engineering
        Estimated Owners: Converted ranges ("20,000 - 50,000") to numerical averages.
        Revenue: Calculated as estimated_num_owners * Price.
        Review Sentiment: Created positive_fraction and negative_fraction from 
        Positive and Negative reviews.
        Tags: One-hot encoded the top 100 tags ("Action", "Indie") using 
        MultiLabelBinarizer.

## Technologies
- Python
- Pandas — Data manipulation and analysis
- NumPy — Numerical computing
- Scikit-learn — Machine learning and data modeling
- Matplotlib — Data visualization
- Seaborn — Statistical data visualization
### Models
- Linear Regression: To predict revenue from Metacritic scores, review fractions, and playtime.
- Tag Analysis: One-hot encoding of game tags to identify genre trends.
- Validation: Train-test split (70% training, 30% testing) with RMSE and R^2 for evaluation.

## Code Snippets
### Data Cleaning & Feature Engineering
Cleaned the raw dataset and created new features, including estimated owners and revenue, for analysis and modeling.
```python
# Remove games without usable sales, price, or Metacritic data
df = df[df['Estimated owners'] != '0 - 0']
df = df[df['Price'] != 0.0]
df = df[df['Metacritic score'] != 0]

# Convert estimated ownership ranges into a numeric value
df['estimated_num_owners'] = (
    df['Estimated owners']
    .str.split(' - ')
    .apply(lambda x: (float(x[0]) + float(x[1])) / 2)
)

# Estimate revenue
df['revenue'] = (
    df['estimated_num_owners'] * df['Price']
).round()
```
### Linear Regression Model
Built and evaluated a linear regression model using game reviews, Metacritic scores, and playtime to predict revenue.
```python
predictors = [
    'Metacritic score',
    'positive_fraction',
    'Average playtime forever'
]

target = 'revenue'

X = df[predictors]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

r2 = reg.score(X_test, y_test)

print(f'R²: {r2:.2f}')
```

## Results

### Revenue vs. Metacritic Scores

A weak positive correlation was found. Higher Metacritic scores generally corresponded with higher revenue, although several outliers were present.

### Predictive Tags

Tags such as **Online Co-Op**, **Building**, and **War** had the strongest positive coefficients in the model. However, many of the highest-revenue games were associated with broader tags such as **Singleplayer**, **Action**, and **Multiplayer**.

### Revenue Prediction

A linear regression model was trained using **Metacritic score, positive review ratio, and average playtime** to predict estimated revenue.

**Model Performance:**

* **RMSE:** 34,475,503.18
* **R²:** 0.27

**Coefficients:**

* Metacritic Score: 650,658.89
* Positive Review Ratio: 1,479,976.01
* Average Playtime: 9,558.03

    

## Discussion
Our analysis found that Metacritic scores, positive reviews, and playtime can help predict game revenue, but they are not sufficient on their own. Genre tags showed little notable correlation with revenue. The model achieved an R² of 0.27, highlighting the limitations of using these variables alone. Among the three predictors, positive review ratio was the strongest.

## Summary
The results partially supported our original hypothesis that Metacritic scores, positive reviews, and genre tags would predict revenue. While genre tags provided little insight, positive reviews, Metacritic scores, and playtime showed stronger relationships with revenue. Future work could improve predictions by incorporating additional game and market data.
