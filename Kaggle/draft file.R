
Y <- as.matrix(PSID$lnwage)
X <- as.matrix(PSID1 %>% select(-lnwage))

full_lasso <- glmnet(
  x = X,
  y = Y,
  alpha = 1
)

lambda.min <- cv.glmnet(X, Y, alpha = 1)$lambda.min
lambda.1se <- cv.glmnet(X, Y, alpha = 1)$lambda.1se

coef(full_lasso, s = lambda.1se)

full_ridge <- glmnet(
  x = X,
  y = Y,
  alpha = 0
)

lambda.min <- cv.glmnet(X, Y, alpha = 0)$lambda.min
lambda.1se <- cv.glmnet(X, Y, alpha = 0)$lambda.1se

coef(full_ridge, s = lambda.1se)



################################################################################
################################################################################

#### Looking for optimal weights

min.mse <- function(data, par){
  with(data, (t(lasso_test_pred * par[1] + en0.5_test_pred * par[2] + ridge_test_pred * par[3] + rf_test_pred * par[4] + xgb_pred * par[5] - lnwage) %*% (lasso_test_pred * par[1] + en0.5_test_pred * par[2] + ridge_test_pred * par[3] + rf_test_pred * par[4] + xgb_pred * par[5] - lnwage) / nrow(data)))
}

# Constraint matrices for optimization
ui <- rbind(c( 1, 0, 0, 0, 0),
            c(-1, 0, 0, 0, 0),
            c( 0, 1, 0, 0, 0),
            c( 0,-1, 0, 0, 0),
            c( 0, 0, 1, 0, 0),
            c( 0, 0,-1, 0, 0),
            c( 0, 0, 0, 1, 0),
            c( 0, 0, 0,-1, 0),
            c( 0, 0, 0, 0, 1),
            c( 0, 0, 0, 0,-1),
            c( 1, 1, 1, 1, 1),
            c(-1,-1,-1,-1,-1))
ci <- c(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 1, -1)

constrOptim(
  theta = c(0.85/3, 0.1/3, 0.05/3, 1/3, 1/3),
  f = min.mse,
  grad = NULL,
  ui = ui,
  ci = ci - 1e-15,
  data = test_preds
)
