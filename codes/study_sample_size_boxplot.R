# R script to generate the boxplot with the used sample sizes in the related work.

# Define the data array
data <- c(25, 15, 5, 8, 5, 105, 22, 6, 14)

# Compute mean
mean_value <- mean(data)

# Define the value to highlight
highlight_value <- 14

# Create the horizontal boxplot
boxplot(data, horizontal=TRUE, main="Distribution of the sample sizes used in related works", 
        col="lightgray", xlab="Sample size values based on related works")

# Add the mean as a point in black
points(mean_value, 1, col="black", pch=19, cex=1.5)
text(mean_value, 1.2, labels=paste("Mean:", round(mean_value,2)), col="black", pos=3)

# Add the highlighted value (14) as a separate point in white with black border
points(highlight_value, 1, col="black", bg="black", pch=17, cex=1.5)
text(highlight_value, 0.8, labels=paste("Our work:", highlight_value), col="black", pos=1)
