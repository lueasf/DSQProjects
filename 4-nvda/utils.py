import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

##############################
# Functions used throughout  #
##############################

def print_output_as_table(output, labels):
    # Get the output index and label
    output_index = output[0].argmax()
    output_label = labels[output[0].argmax()]

    # Create a dataframe
    df = pd.DataFrame({
        'Output index': [output_index],
        'Output label': [output_label]
    })

    # Return the dataframe
    return df


##############################
# Functions used throughout  #
##############################

def plot1(x, y, m, b):
    # this is all plotting stuff. You don't have to understand it.
    with plt.style.context('bmh'):
        plt.plot(x, y, alpha=0.5, label="Prediction Line")
        xmax = 10
        ymax = 10
        plt.axhline(
            y=2,
            xmin=(2 / xmax),
            xmax=(6 / xmax),
            color="red",
            linestyle="--",
            lw=5,
            label="slope (horizontal component)",
        )
        plt.axvline(
            x=6,
            ymin=(2 / ymax),
            ymax=(5 / ymax),
            color="red",
            linestyle="--",
            lw=5,
            label="slope (vertical component)",
        )
        plt.text(
            4,
            1.5,
            f"+4 over",
            color="red",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            alpha=0.9,
            zorder=20,
        )
        plt.text(
            6.2,
            3,
            f"+3 up",
            color="red",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            alpha=0.9,
            zorder=20,
        )
        plt.scatter(
            x=[0],
            y=b,
            color="red",
            marker="X",
            s=200,
            edgecolors="black",
            label="Intercept",
            alpha=0.75,
            zorder=20,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)
        plt.title(f"y = {m}x + {b}")
        plt.legend()
        plt.grid("on")
        plt.show()
    

def plot2(X, Y, slope, intercept, new_x):
    new_y = slope * new_x + intercept 
    regression_line = slope * X + intercept 
    with plt.style.context('bmh'):
        plt.scatter(X, Y, color="blue", alpha=0.5, zorder=0)
        plt.scatter(
            new_x,
            new_y,
            color="red",
            marker="X",
            s=200,
            edgecolors="black",
            label="New Point",
            alpha=0.75,
            zorder=20,
        )
        plt.plot(
            X,
            regression_line,
            color="green",
            label=f"Linear Regression (y = {slope:.2f}x + {intercept:.2f})",
            alpha=0.5,
            zorder=10,
        )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Generated Data")
        plt.legend()
        plt.show()
    return new_y
    
def parse_to_dict(data):
    # decode byte string to normal string and split lines
    lines = data.decode().split('\n')

    # remove the empty lines
    lines = [line for line in lines if line]

    # the first line is the header
    header = lines[0].split()

    result = []
    users = set()
    # process rest of the lines
    for line in lines[1:]:
        line_dict = {}
        # split the line into parts
        parts = line.split()

        # process each part of the header
        for i, part in enumerate(header):
            if i < len(parts):
                line_dict[part] = parts[i]
                if part == "USER":
                    users.add(parts[i])
            else:
                line_dict[part] = None
        result.append(line_dict)

    return result, float(len(result)), float(len(users))