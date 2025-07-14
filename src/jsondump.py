import json

seed = 100
best_params = {'n_layers': 1, 'n_units_l0': 128, 'lr': 0.0035767968127597776, 'optimizer': 'Adam', 'dropout_rate': 0.23854263324563796}
best_val_loss = 0.07849778819084167   
test_loss = 0.0723                    
test_accuracy = 0.9775              

summary = {
    "seed": seed,
    "best_val_loss": best_val_loss,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "best_params": best_params
}

# Save to JSON file
with open(f"summary_seed_{seed}.json", "w") as f:
    json.dump(summary, f, indent=4)

print(f"Saved full summary for seed {seed}.")
