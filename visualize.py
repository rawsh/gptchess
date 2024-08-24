import matplotlib.pyplot as plt

# Model names
models = [
    "gpt-4o-mini [zero shot]",
    "gpt-4o-mini",
    "gpt-4o-mini (self correct)",
    "gpt-4o-mini-ft",
    "gpt-4o-mini-ft (self correct)",
    "gpt-4o-mini-ft2 (4o teacher)",
    "gpt-4o [zero shot]",
    "gpt-4o",
    "gpt-4o (self correct)",
    "gpt-4o-ft",
    "gpt-4o-ft (self correct)",
    "babbage-002",
    "davinci-002",
    "davinci-002-ft"
]

# Accuracies for each model
accuracies = [17.4, 25.99, 31.72, 57.71, 65.64, 54.19, 28.44, 55.07, 63.88, 58.59, 71.37, 61.23, 74.45, 76.21]

# Create a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(models, accuracies)  # Use scatter instead of bar

# Add labels and title
plt.xlabel("model")
plt.ylabel("accuracy (%)")
plt.title("llm pgn chess puzzle solving")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Tight layout to prevent overlapping elements
plt.tight_layout()

# You can optionally save or display the plot
plt.savefig("model_accuracy.png") 
plt.show()