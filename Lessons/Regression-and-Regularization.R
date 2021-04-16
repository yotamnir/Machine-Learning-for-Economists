########################################################
##
##      Lesson 5: Regression and Regularization
##
########################################################

## Loading packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,   # for data wrangling and visualization
  knitr,       # for displaying nice tables
  broom,       # for tidying estimation output
  here,        # for referencing folders and files
  glmnet,      # for estimating lasso and ridge
  gamlr,       # for forward stepwise selection
  pls,         # for estimating PCR and PLS
  elasticnet,  # for estimating PCR and PLS
  ggfortify
)

## Loading Browser data (12.04.2021)
urlfile <- "https://raw.githubusercontent.com/ml4econ/lecture-notes-2021/master/05-regression-regularization/data/browser-all.csv"
browser <- read.csv(url(urlfile))

Y_browser <- browser_mat[, 1]     # response
X_browser <- browser_mat[, 2:201] # features

## Useful to transform to matrix
browser_mat <- browser %>% 
  as.matrix()

## OLS all variables
lm_fit <- lm(log_spend ~ ., data = browser)
## Show top three most significant
lm_fit %>% 
  tidy() %>% 
  arrange(p.value) %>% 
  head(3) %>% 
  kable(format = "simple", digits = 2)

## Show mean squared error (clearly an underestimate because we are overfitting)
lm_fit %>% 
  augment() %>%
  summarise(mse = mean((log_spend - .fitted)^2)) %>% 
  kable(format = "simple", digits = 3)

## Subset selection: forward stepwise algorithm
fit_step <- gamlr(X_browser, Y_browser, gamma=Inf, lmr=.1)
plot(fit_step, df=FALSE, select=FALSE)

## Estimating ridge (note: this is actually an elastic ridge estimation, so when alpha = 0 we have a ridge estimation, and when alpha = 1 we have a lasso estimation)
fit_ridge <- glmnet(
  x = X_browser,
  y = Y_browser,
  alpha = 0
)
plot(fit_ridge, xvar = "lambda")

## Tuning lambda (cross validation - default is 10 folds)
cv_ridge <- cv.glmnet(x = X_browser, y = Y_browser, alpha = 0)
plot(cv_ridge)

## Estimating lasso (just changing alpha to 1)
fit_lasso <- glmnet(
  x = X_browser,
  y = Y_browser,
  alpha = 0
)
plot(fit_lasso, xvar = "lambda")

## Tuning lambda (exactly the same as before, with alpha = 1)
cv_lasso <- cv.glmnet(x = X_browser, y = Y_browser, alpha = 1, nfolds = 10)
plot(cv_lasso, xvar = "lambda")

## Showing the coefficients selected by the lasso model
  ## Using the minimal lambda
coef(cv_lasso, s = "lambda.min") %>%
  tidy() %>%
  as_tibble()
  ## Using the lambda one SE to the right
coef(cv_lasso, s = "lambda.1se") %>%
  tidy() %>%
  as_tibble()

####################################################################
# Dimensionality Reduction (Principal components regression - PCR)
####################################################################

## Loading data (15.04.2021)
urlfile <- "https://raw.githubusercontent.com/ml4econ/lecture-notes-2021/master/05-regression-regularization/data/nbc_showdetails.csv"
shows <- read.csv(url(urlfile))
urlfile <- "https://raw.githubusercontent.com/ml4econ/lecture-notes-2021/master/05-regression-regularization/data/nbc_pilotsurvey.csv"
survey <- read.csv(url(urlfile))

## Grouping by show rather than by viewer
survey_mean <- survey %>% 
  select(-Viewer) %>% 
  group_by(Show) %>% 
  summarise_all(list(mean)) # creates by-show mean for all rows

## Joining the two data sets
df_join <- shows %>% 
  left_join(survey_mean) %>% 
  select(PE, starts_with("Q"), Genre, Show)

## Creating the data matrix for the Principal Component
X <- df_join %>% 
  select(starts_with("Q")) %>% 
  as.matrix()

## Principal components based on x (scaled)
PC <- X %>% 
  prcomp(scale. = TRUE) %>%
  predict()

## Creating an object that is just the outcome variable
Y <- df_join %>% 
  select(PE) %>% 
  as.matrix()

## Running PCR using the pls package (returns number of PCs = number of x's - 20 in this case)
cv_pcr <- pcr(
  Y ~ X,       # recall Y and X are matrices
  scale = TRUE,
  validation = "CV",
  segments = 10     # i.e. folds  
)
validationplot(cv_pcr)

## Observing the PCR lambdas (called loadings matrix or rotation matrix)
pca <- X %>% prcomp(scale. = TRUE)
pca$rotation[, 1:2] %>% round(1)  # looking only at the first two components

## Graph of the variance explained by each PC
pca %>% 
  tidy("pcs") %>% 
  ggplot(aes(PC, percent)) +
  geom_line() +
  geom_point() +
  labs(
    y = "percent of explaind variance",
    x = "principal component number"
  )

## Not entirely clear but somthing like loadings matrix in graph form
pca %>% 
  autoplot(
    data = df_join,
    colour = "Genre",
    size = 3
  ) +
  theme(legend.position = "top")

## Sparse PCA (SPCA) - employs lasso-type penalties on the PCs (estimated here using the spca() function from the elasticnet package)
spc <- spca(
  x = X,
  K = 2,
  type = "predictor",
  sparse = "varnum",
  para = c(10, 10)
)
spc$loadings %>% round(1)

## Running Partial Least Squares (PLS)
cv_pls <- plsr(
  Y ~ X,
  scale = TRUE,
  validation = "CV",
  segments = 10       
)
validationplot(cv_pls)
