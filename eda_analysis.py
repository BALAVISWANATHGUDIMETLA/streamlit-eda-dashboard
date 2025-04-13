import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the log file
log_path = "lstm_warm_start_log.txt"
df = pd.read_csv(log_path, sep="\t")

# Basic summary
print("🔹 Data Head:\n", df.head())
print("\n🔹 Data Description:\n", df.describe())

# Add rolling average and loss delta
df['Rolling_Loss'] = df['Loss'].rolling(window=5).mean()
df['Loss_Diff'] = df['Loss'].diff()

# Find epoch with minimum loss
min_loss_row = df.loc[df['Loss'].idxmin()]
print(f"\n🔻 Lowest Loss: {min_loss_row['Loss']:.4f} at Epoch {int(min_loss_row['Epoch'])}")

# Line Plot: Loss and Rolling Avg
plt.figure(figsize=(12, 6))
plt.plot(df['Epoch'], df['Loss'], label='Loss', marker='o')
plt.plot(df['Epoch'], df['Rolling_Loss'], label='Rolling Avg (5)', linestyle='--')
plt.scatter(min_loss_row['Epoch'], min_loss_row['Loss'], color='red', label='Min Loss', zorder=5)
plt.text(min_loss_row['Epoch'], min_loss_row['Loss'] + 0.01,
         f"Epoch {int(min_loss_row['Epoch'])}\nLoss: {min_loss_row['Loss']:.4f}",
         color='red', fontsize=9)
plt.title("Loss and Rolling Average Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_with_rolling_avg.png")
plt.show()

# Bar Plot: Loss Delta
plt.figure(figsize=(12, 5))
plt.bar(df['Epoch'][1:], df['Loss_Diff'][1:], color='orange')
plt.axhline(0, color='black', linestyle='--')
plt.title("Loss Change Between Epochs")
plt.xlabel("Epoch")
plt.ylabel("Δ Loss")
plt.tight_layout()
plt.savefig("loss_difference.png")
plt.show()

# Histogram: Distribution of Loss
plt.figure(figsize=(8, 5))
plt.hist(df['Loss'], bins=10, color='purple', edgecolor='black')
plt.title("Distribution of Loss Values")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("loss_histogram.png")
plt.show()

# Zoom-in Plot: Last 10 Epochs
plt.figure(figsize=(10, 4))
df_tail = df.tail(10)
plt.plot(df_tail['Epoch'], df_tail['Loss'], marker='o', linestyle='-', label='Last 10 Epochs')
plt.title("Zoomed-In View: Last 10 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("zoomed_loss_last10.png")
plt.show()

# Convergence Velocity Plot (Absolute Loss Change)
plt.figure(figsize=(10, 4))
plt.plot(df['Epoch'][1:], df['Loss_Diff'][1:].abs(), label='|Δ Loss|', color='teal')
plt.title("Convergence Velocity (|Δ Loss| per Epoch)")
plt.xlabel("Epoch")
plt.ylabel("Absolute Δ Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("convergence_velocity.png")
plt.show()

# Correlation Heatmap (if more columns are added in future)
if df.shape[1] > 3:
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Between Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()