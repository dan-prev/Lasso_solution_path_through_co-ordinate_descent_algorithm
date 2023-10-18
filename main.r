library(glmnet)

lamda_range<-exp(seq(0*log(10), 3*log(10), length_100))

X <- as.matrix(X_simulated)
y <- as.vector(unlist(y_simulated))

fit <- glmnet(X, y, alpha=1, lambda=lambda_range)

beta_glmnet<-as.matrix(coef(fit))
print(beta_glmnet)
write.csv(beta_glmnet, "beta_glmnet.csv", row.names=FALSE)