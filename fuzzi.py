import tensorflow as tf
import numpy as np

# ==== DEFINISI FUZZY SET ====

def trapmf(x, a, b, c, d):
    """Fungsi keanggotaan trapesium"""
    return tf.clip_by_value(tf.minimum(
        tf.minimum((x - a) / (b - a + 1e-6), 1.0),
        tf.minimum((d - x) / (d - c + 1e-6), 1.0)
    ), 0, 1)

def trimf(x, a, b, c):
    """Fungsi keanggotaan segitiga"""
    return tf.clip_by_value(tf.minimum((x - a)/(b - a + 1e-6), (c - x)/(c - b + 1e-6)), 0, 1)

# ==== USER: DEFINISI VARIABLE & FUZZY SET ====

fuzzy_input = {
    'suhu': {
        'dingin': lambda x: trapmf(x, 0, 0, 20, 30),
        'normal': lambda x: trimf(x, 20, 30, 40),
        'panas':  lambda x: trapmf(x, 30, 40, 60, 60),
    },
    'kelembapan': {
        'rendah': lambda x: trapmf(x, 0, 0, 30, 50),
        'sedang': lambda x: trimf(x, 30, 50, 70),
        'tinggi': lambda x: trapmf(x, 60, 70, 100, 100),
    }
}

fuzzy_output = {
    'kipas': {
        'lambat': lambda x: trapmf(x, 0, 0, 20, 40),
        'sedang': lambda x: trimf(x, 30, 50, 70),
        'cepat':  lambda x: trapmf(x, 60, 80, 100, 100),
    }
}

# ==== USER: ATURAN FUZZY ====

rules = [
    # (IF kondisi, THEN hasil)
    ( {'suhu': 'dingin', 'kelembapan': 'rendah'}, {'kipas': 'lambat'} ),
    ( {'suhu': 'normal', 'kelembapan': 'sedang'}, {'kipas': 'sedang'} ),
    ( {'suhu': 'panas', 'kelembapan': 'tinggi'}, {'kipas': 'cepat'} ),
    ( {'suhu': 'panas', 'kelembapan': 'sedang'}, {'kipas': 'cepat'} ),
]

# ==== FUNGSIONAL ====

def fuzzifikasi(inputs):
    μ = {}
    for var_name, var_value in inputs.items():
        μ[var_name] = {}
        for label, func in fuzzy_input[var_name].items():
            μ[var_name][label] = func(tf.constant(var_value, dtype=tf.float32))
    return μ

def inferensi(fuzzy_values, output_name='kipas'):
    x_output = tf.linspace(0.0, 100.0, 1000)
    hasil_rules = []

    for kondisi, hasil in rules:
        # Ambil nilai min dari semua kondisi (AND fuzzy)
        values = []
        for var, label in kondisi.items():
            values.append(fuzzy_values[var][label])
        α = tf.reduce_min(tf.stack(values))  # derajat aturan

        # Ambil fungsi keanggotaan hasil output
        for out_var, out_label in hasil.items():
            μ_output = fuzzy_output[out_var][out_label](x_output)
            hasil_rules.append(tf.minimum(α, μ_output))

    # Gabungkan semua output (OR fuzzy)
    output_agregat = tf.reduce_max(tf.stack(hasil_rules), axis=0)
    return x_output, output_agregat

def defuzzifikasi(x, μ):
    return tf.reduce_sum(x * μ) / (tf.reduce_sum(μ) + 1e-6)

# ==== MAIN FUNGSI ====

def fuzzy_kontrol(input_user: dict, output_name='kipas'):
    μ_input = fuzzifikasi(input_user)
    x_out, μ_out = inferensi(μ_input, output_name=output_name)
    crisp = defuzzifikasi(x_out, μ_out)
    return crisp.numpy()

# ==== CONTOH PENGGUNAAN ====

input_user = {
    'suhu': 35,
    'kelembapan': 60
}

output = fuzzy_kontrol(input_user)
print(f"Input: {input_user} → Kecepatan kipas: {output:.2f}%")
