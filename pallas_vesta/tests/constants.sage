# pasta_params.sage

def get_montgomery_parameters(p, generator, n_bits):
    R = 2^n_bits
    R_mod_p = R % p
    R2_mod_p = (R * R) % p
    R3_mod_p = (R2_mod_p * R) % p
    inv = (-inverse_mod(p, 2^64)) % 2^64

    # Find the largest power-of-two root of unity
    two_adicity = (p - 1).valuation(2)
    root_of_unity = None
    for g in range(2, p):
        candidate = Mod(g, p)
        if pow(candidate, (p - 1) // 2, p) != 1:
            root = candidate ^ ((p - 1) // (2^two_adicity))
            if pow(root, 2^two_adicity, p) == 1:
                root_of_unity = int(root)
                break

    delta = int(generator)

    return {
        "modulus": int(p),
        "inv": int(inv),
        "R": int(R_mod_p),
        "R2": int(R2_mod_p),
        "R3": int(R3_mod_p),
        "Generator": int(generator),
        "Root_of_Unity": root_of_unity,
        "Delta": delta
    }

# --- Parameters from halo2 source (same as Zcash's Pasta curves) ---
# See: https://github.com/zcash/halo2/blob/main/halo2curves/src/pasta/mod.rs

# Pallas is the base field of Vesta
p_pallas = 28948022309329048855892746252171976963363056481941647379679742748393362948097
# Vesta is the base field of Pallas
p_vesta  = 28948022309329048855892746252171976963363056481941647379679742748393362948099

# Multiplicative generators from halo2's constants (already known for Pasta curves)
g_pallas = 5
g_vesta = 5

params_pallas = get_montgomery_parameters(p_pallas, g_pallas, 256)
params_vesta  = get_montgomery_parameters(p_vesta, g_vesta, 256)

import json

print("Pallas Parameters:")
print(json.dumps(params_pallas, indent=4))
print("\nVesta Parameters:")
print(json.dumps(params_vesta, indent=4))
