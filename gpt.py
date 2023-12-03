#!/usr/bin/env python

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Define lattice size
rows = 40
columns = 280

# Define the special atoms' coordinates
special_atoms = [(140, 20), (160, 20)]

# Aesthetic colors
blue_color = "#2a6fb0"
black_color = "black"

# Create the figure and axis
fig_width_inches = 3.375  # Width for a 2-column A4 page in inches
fig_height_inches = fig_width_inches * (rows / columns)
fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
ax.set_aspect("equal")

# Loop through the lattice coordinates and draw colored dots
for x in range(columns):
    for y in range(rows):
        if (x, y) in special_atoms:
            color = black_color
            size = 20
        else:
            color = blue_color
            size = 0.08  # Further reduced the size of blue dots
        ax.plot(x, y, "o", color=color, markersize=size)

# Draw arrows and labels for the black atoms
arrow_style = mpatches.ArrowStyle.Fancy(head_length=0.4, head_width=0.4, tail_width=0.2)
arrow_props = dict(arrowstyle=arrow_style, facecolor=black_color, edgecolor=black_color)

arrow1 = plt.annotate(
    "",
    xy=(special_atoms[0][0], special_atoms[0][1] + 6),
    xytext=(special_atoms[0][0], special_atoms[0][1]),
    arrowprops=arrow_props,
)
plt.text(
    special_atoms[0][0] - 1.5,
    special_atoms[0][1] + 7,
    r"$\mathbf{S}_1$",
    fontsize=12,
    color=black_color,
)

arrow2 = plt.annotate(
    "",
    xy=(special_atoms[1][0], special_atoms[1][1] - 6),
    xytext=(special_atoms[1][0], special_atoms[1][1]),
    arrowprops=arrow_props,
)
plt.text(
    special_atoms[1][0] - 1.5,
    special_atoms[1][1] - 9,
    r"$\mathbf{S}_2$",
    fontsize=12,
    color=black_color,
)

# Set axis limits and remove ticks
ax.set_xlim(-1, columns)
ax.set_ylim(-1, rows)
ax.set_xticks([])
ax.set_yticks([])

# Save the figure as an image
plt.savefig("lattice_plot.pdf", dpi=1200, bbox_inches="tight")
