import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Lade beide Bilder
img1 = mpimg.imread('plot_1_arbitrage_per_mwh.png')
img2 = mpimg.imread('plot_2_cycles.png')

# Erstelle eine Figure mit zwei Subplots nebeneinander, exakt wie die Einzelplots (je 5x3 inch)
fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=300)
axes[0].imshow(img1)
axes[0].axis('off')
axes[1].imshow(img2)
axes[1].axis('off')

plt.tight_layout(pad=0.2)
plt.savefig('combined_arbitrage_cycles_exact.png', dpi=300, bbox_inches='tight')
plt.close()
print('combined_arbitrage_cycles_exact.png erstellt.')
