import matplotlib.pyplot as plt
d = {
    "LoRA": [43.0, 43.6, 44.2, 45.5, 46.5, 47.1],
    "adapter": [43.3, 44.2, 45.2, 46.0, 46.6, 46.7]
}
for peft, ys in d.items():
    xs = list(range(1, len(ys)+1))
    color = "r" if peft == "LoRA" else "b"
    plt.plot(xs, ys, marker= "o", linestyle='-', color = color, label = peft)
plt.xticks(xs, ["8", "32", "64", "128", "256", "512"])
plt.xlabel('LoRA rank/adapter size')
plt.ylabel('RougeL Score')
plt.legend()
plt.savefig("trainable_parameter.pdf")