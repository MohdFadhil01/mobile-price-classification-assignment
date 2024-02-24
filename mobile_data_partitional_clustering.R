# Load the Mobile Price Classification dataset
mobile_data <- read.csv("train.csv")

# Relevant attributes for clustering, including the target variable
clustering_data <- mobile_data[, c('battery_power', 'ram', 'int_memory', 'price_range')]

# Extract the target variable for later evaluation
target_variable <- clustering_data$price_range
clustering_data <- clustering_data[, -which(names(clustering_data) == 'price_range')]

# Elbow method to determine optimal K values
wss <- numeric(10)

for (i in 1:10) {
  kmeans_model <- kmeans(clustering_data, centers = i, nstart = 10)
  wss[i] <- kmeans_model$tot.withinss
}

# Plot the elbow method
plot(1:10, wss, type = "b", main = "Elbow Method", xlab = "Number of Clusters", ylab = "Within Sum of Squares")

# Apply k-means clustering for k=2
kmeans_result_2 <- kmeans(clustering_data, centers = 2, nstart = 10)

# Apply k-means clustering for k=3
kmeans_result_3 <- kmeans(clustering_data, centers = 3, nstart = 10)

# Apply k-means clustering for k=4
kmeans_result_4 <- kmeans(clustering_data, centers = 4, nstart = 10)

# Visualize clustering results for k=2
pairs(clustering_data, col = kmeans_result_2$cluster, pch = 20, main = "K-means Clustering (k=2)")

# Visualize clustering results for k=3
pairs(clustering_data, col = kmeans_result_3$cluster, pch = 20, main = "K-means Clustering (k=3)")

# Visualize clustering results for k=4
pairs(clustering_data, col = kmeans_result_4$cluster, pch = 20, main = "K-means Clustering (k=4)")

# Selecting numeric columns for scaling
numeric_columns <- c('battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time')

# Create a scaled version of the numeric columns
mobile_data_scaled <- scale(mobile_data[, numeric_columns])

# Get cluster assignments
cluster_assignments <- kmeans_result_2$cluster

# Calculate Silhouette Score manually
silhouette_scores <- silhouette(cluster_assignments, dist(mobile_data_scaled))
silhouette_avg <- mean(silhouette_scores[, "sil_width"])
cat("Silhouette Score:", silhouette_avg, "\n")

# Calculate Within-Cluster Sum of Squares (WCSS)
wcss <- sum(kmeans_result_2$withinss)
cat("Within-Cluster Sum of Squares (WCSS):", wcss, "\n")
