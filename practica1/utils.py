"""
Utility helpers used across the project.
"""

# =========================
# Printing / formatting
# =========================

def print_results(labels, values):
    """ Print labeled results in a structured format. """
    assert len(labels) == len(values)
    idx = 0
    for i in range(len(labels)):
        if labels[i] == "results":
            idx = i + 1
            break
        print("-" * 60)
        print(labels[i], "=")
        print(values[i])
        print()
    
    print("="*10, "RESULTS", "="*12)

    for i in range(idx, len(labels)):
        print(f"{labels[i]} = {values[i]}")