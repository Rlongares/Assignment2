

data <- read.table(file="dataset_es_dev.json")

data <- subset( data, select = -c(V1, V3,V4,V5,V7,V8,V9,V11,V12,V13,V15,V16,V17,V19,V20,V21,V23,V24,V25,V26,V27,V28,V29,V31) )

colnames(data) = c("review_id", "product_id",
                   "reviewer_id","stars",
                   "review_body","review_title", "product_category")


data <- data %>%
  add_column(Class = if_else(.$stars > 3,"positive", "negative"))
data$Class[data$stars == 3] <- "neutral"

write.csv(data,"Areviews.csv", row.names = TRUE)

