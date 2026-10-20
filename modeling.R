
# Week 3 Task: Statistical Analysis and Predictive Modeling using R
# Focus: Hypothesis Testing, Logistic Regression, Diagnostics & ROC-AUC


library(dplyr)
library(ggplot2)

# 1. Dataset Initialization
titanic_data <- data.frame(
  PassengerId = 1:20,
  Survived = c(0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1),
  Pclass = c(3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3),
  Sex = c("male", "female", "female", "female", "male", "male", "male", "male", 
          "female", "female", "female", "female", "male", "male", "female", 
          "female", "male", "male", "female", "female"),
  Age = c(22, 38, 26, 35, 35, 28, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 2, 28, 31, 28),
  Fare = c(7.25, 71.28, 7.92, 53.1, 8.05, 8.45, 51.86, 21.07, 11.13, 30.07, 
           16.7, 26.55, 8.05, 31.27, 7.85, 16.0, 29.12, 13.0, 18.0, 65.0),
  stringsAsFactors = FALSE
)

# Convert Factors
titanic_data$Survived_Factor <- factor(titanic_data$Survived, levels = c(0, 1), labels = c("No", "Yes"))
titanic_data$Pclass <- factor(titanic_data$Pclass, levels = c(1, 2, 3), labels = c("1st", "2nd", "3rd"))
titanic_data$Sex <- factor(titanic_data$Sex)

# 2. Exploratory Statistical Analysis & Hypothesis Testing
cat("=== 1. Chi-Square Test of Independence (Gender vs Survival) ===\n")
chisq_res <- chisq.test(table(titanic_data$Sex, titanic_data$Survived_Factor))
print(chisq_res)

cat("\n=== 2. Two-Sample t-Test (Age across Survival Outcomes) ===\n")
t_res <- t.test(Age ~ Survived_Factor, data = titanic_data)
print(t_res)

# 3. Model Building: Binary Logistic Regression
set.seed(42)
train_idx <- sample(1:nrow(titanic_data), size = 0.75 * nrow(titanic_data))
train_set <- titanic_data[train_idx, ]
test_set  <- titanic_data[-train_idx, ]

logit_model <- glm(Survived ~ Sex + Pclass + Age, data = train_set, family = binomial(link = "logit"))
cat("\n=== 3. Logistic Regression Summary ===\n")
print(summary(logit_model))

# Odds Ratios
cat("\n=== 4. Odds Ratios (exp(Beta)) ===\n")
print(exp(coef(logit_model)))

# 4. Model Predictions & Confusion Matrix Evaluation
pred_probs <- predict(logit_model, newdata = test_set, type = "response")
pred_classes <- ifelse(pred_probs > 0.5, 1, 0)
cm <- table(Actual = test_set$Survived, Predicted = pred_classes)
cat("\n=== 5. Confusion Matrix on Test Set ===\n")
print(cm)

# 5. Diagnostic Plot: Model Feature Coefficients (Odds Ratios)
odds_df <- data.frame(
  Feature = names(coef(logit_model))[-1],
  OddsRatio = exp(coef(logit_model))[-1]
)

p_odds <- ggplot(odds_df, aes(x = reorder(Feature, OddsRatio), y = OddsRatio)) +
  geom_col(fill = "#2980B9", width = 0.5) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Feature Impact on Survival Odds (Odds Ratios)",
       subtitle = "Values < 1 indicate lower probability of survival compared to baseline",
       x = "Model Covariates", y = "Odds Ratio (Baseline = 1.0)") +
  theme_minimal()

print(p_odds)
