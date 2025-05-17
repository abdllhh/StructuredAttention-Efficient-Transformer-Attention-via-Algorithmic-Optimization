import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('attention_benchmark_results.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Execution Time vs Sequence Length
ax1.plot(df['sequence_length'], df['standard_attention_ms'], 'o-', label='Standard Attention', color='blue')
ax1.plot(df['sequence_length'], df['kd_tree_attention_ms'], 's-', label='KD-Tree Attention', color='red')
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Attention Mechanism Performance Comparison')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add best-fit lines to show complexity trends
x = df['sequence_length'].values
# For standard attention (should be O(n²))
y_std = df['standard_attention_ms'].values
p_std = np.polyfit(x, y_std, 2)  # Quadratic fit for standard attention
x_smooth = np.linspace(min(x), max(x), 100)
y_std_smooth = np.polyval(p_std, x_smooth)
ax1.plot(x_smooth, y_std_smooth, '--', color='lightblue', alpha=0.7, label='O(n²) trend')

# For KD-Tree attention (should be closer to O(n log n))
y_kd = df['kd_tree_attention_ms'].values
p_kd = np.polyfit(x, y_kd, 1)  # Linear fit for KD-Tree (simplified)
y_kd_smooth = np.polyval(p_kd, x_smooth)
ax1.plot(x_smooth, y_kd_smooth, '--', color='lightcoral', alpha=0.7, label='O(n) trend')

ax1.legend()

# Plot 2: Speedup vs Sequence Length
ax2.plot(df['sequence_length'], df['speedup'], 'o-', color='green')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)  # Reference line at speedup = 1
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Speedup (Standard / KD-Tree)')
ax2.set_title('Speedup Factor by Sequence Length')
ax2.grid(True, alpha=0.3)

# Add annotations
for i, row in df.iterrows():
    ax2.annotate(f"{row['speedup']:.2f}x",
                 (row['sequence_length'], row['speedup']),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')



plt.tight_layout()
plt.savefig('attention_performance.png', dpi=300)
plt.show()

print("Plot saved as 'attention_performance.png'")