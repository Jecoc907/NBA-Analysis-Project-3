# NBA-Analysis-Project-3

## Introduction
In this assignment, we aim to predict the Player Efficiency Rating (PER) of NBA basketball players using their in-game statistics, including points, turnovers, and field goals, etc. As a follow-up report to HW3, we enhance our Decision Tree Regression models by introducing ensemble algorithms such as bagging, Random Forests, and XGBoosting. The dataset remains the same from HW3, containing in-game statistics of 539 NBA players in the 2022-2023 season with 52 columns (we combined two player stats tables: 'total' and 'advanced'). The data was obtained from basketball-reference.com (2022-23 NBA Player Stats: Totals: https://www.basketball-reference.com/leagues/NBA_2023_totals.html /
Advanced: https://www.basketball-reference.com/leagues/NBA_2023_advanced.html ).

## Preliminary Analysis
Upon uploading our dataset to Google Colab, we began by analyzing the basics of our dependent variable. Using the 'describe()' command, we found that our response variable, 'PER', averaged at 13.32 with the lowest at -20.90 (Alondes Williams) and the highest at 65.60 (Stanley Umude). The output of the code is attached.

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/493a5e2f-1ef8-4942-9038-639aa9e9e715)

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/1591f3c0-e82d-4ce4-a248-1c56a4630124)

## Data Processing (a different approach was used to deal with Null Values)
Before running our Decision Tree Regression model, we encountered two main problems: string columns and NaN values. Using the 'dtypes' command allowed us to identify the string columns, such as player names, positions, and teams, which should be excluded from our model.

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/bc394789-602c-4d25-9cb8-afb4565ab153)

Next, we noticed a few NaN values in our dataset, which obstructed us from training and testing our model. Upon investigation, we found that all of them represent division by zero errors. For example, we have 16 observations (players) with NaN values in '3P%', representing three-point field goal percentage, because none of them attempted a single three-point shot during the season. Instead of replacing NaN with zeros as in HW3, we opted for imputation (replacing NaN values with mean or median) to optimize the machine learning process for higher model accuracy and performance.

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/08dd6457-3516-419e-b70b-1e04094c7818)

## Decision Tree Model Building
After data processing, we began building our decision tree regressor model. With 46 features after dropping the dependent variable 'PER', the string columns, and the 'rank' column, our model achieved a high R-squared of 0.86 with a low MSE of 3.29 using the validation-set method. However, using more than 40 features in the model could lead to overfitting. To mitigate this, we employed pruning techniques by tuning the model’s hyperparameters (max_depth, min_sample_split, and min_samples_leaf) using GridSearchCV with 5-fold cross-validation. With the best parameters of max_depth = 7, min_samples_leaf = 4, and min_samples_split = 10, we ultimately built a model with 23 features which achieved a score of 0.869.

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/0bf834f8-11b4-405d-8368-c854c7732c8b)

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/a7383a1b-c28e-4c07-9f84-7355dc03eb3a)

In the appendix (2), we presented all the variables in the model along with their corresponding feature importance. Among the 14 variables included in the model, we learned that 'WS/48' is the most important factor in determining player efficiency, followed by 'OBPM', 'BPM', 'FG%', etc. One explanation could be that players with higher wins shares per 48 minutes (WS/48 > 0.113) and higher Box Plus/Minus (BPM > 17.4) are more efficient players. Although this smaller, pruned tree produces greater error, we assume it will perform better and more consistently across different datasets.

## Ensemble Algorithms (Bootstrapping, XGboosting, Random Forrest)
To further reduce the variance of our model, we decided to use ensemble algorithms (bagging, XGBoosting, and random forest). After fitting the model to the training data and evaluating the models on the test data using the '.score' function, we found that all three ensemble algorithms outperformed our previous model with an average score of 0.95 compared to 0.869 (old model). This confirmed our decision to optimize our model with one of the aforementioned ensemble algorithms.

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/c80b41a0-8c28-4c27-b2cb-75e7c9820095)

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/b31079bf-8843-4938-9399-abea18caddea)

![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/d6653f3d-1f03-489b-b299-c13ed71f849c)

After rigorous experimentation and hyperparameter tuning using GridSearchCV, we assessed the performance of three ensemble algorithms: Bagging, XGBoosting, and Random Forest. Each showed promising results in predicting the player efficiency rating of NBA players based on their in-game statistics. In terms of MSE, XGBoosting achieved the lowest MSE (0.66), indicating superior predictive performance compared to the others (Random Forest: 1.14; Bagging: 0.99). Given its prediction performance and gradient-based learning concept (self-adjusted for overfitting), we suggest deploying the XGBoost regressor model for predicting NBA player efficiency. We acknowledge that XGBoost has some general problems, such as low interpretability, high tuning complexity, and computational burden compared to the other two models. Some directions for improvement will be covered in the following section.!

## Limitations
We acknowledge several limitations in our analysis. Firstly, the interpretability of our model remains a challenge. XGBoost is more complex and less interpretable because it introduces gradient-based learning and uses a more sophisticated method for selecting splits compared to simpler models like basic decision trees and linear regression. Secondly, the size of our dataset, limited to the 2023-24 season with data from 539 players, may affect the accuracy of our model. For instance, some players may have only played a few games in the 2023-24 season. Expanding the dataset to include player stats from the past decade could enhance model performance.

## Appendix
### Appendix(1): Original Model Demonstration
![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/f3100c90-968b-4594-a505-9817b98f1b5d)

### Appendix(2): Features Included
![image](https://github.com/Jecoc907/NBA-Analysis-Project-3/assets/71363412/ba0d35c5-e7db-4e03-be69-f2386cd1fc8a)










