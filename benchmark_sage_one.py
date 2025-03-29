import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sage_one import SAGE_ONE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import os
import hashlib

seed = 42


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seeds()


def generate_parity_levels(n_samples=10000, seq_length=10):
    X = np.random.randint(0, 3, size=(n_samples, seq_length))
    # X = np.random.choice([0, 1, 2], size=(n_samples, seq_length))

    # Nível 1: Paridade simples
    y_lvl1 = (np.sum(X, axis=1) % 2).reshape(-1, 1).astype(np.float32)

    # Nível 2: Paridade apenas nas posições pares
    y_lvl2 = (np.sum(X[:, ::2], axis=1) % 2).reshape(-1, 1).astype(np.float32)

    # Nível 3: Soma ponderada nas ímpares com pesos 2^pos
    weights = 2 ** np.arange(1, seq_length, 2)
    weighted_sum = np.sum(X[:, 1::2] * weights, axis=1)
    y_lvl3 = (weighted_sum % 2).reshape(-1, 1).astype(np.float32)

    # Nível 4: Paridade da soma entre XORs das posições alternadas
    xor_pairs = X[:, :-1:2] ^ X[:, 1::2]
    parity_xor = (np.sum(xor_pairs, axis=1) %
                  2).reshape(-1, 1).astype(np.float32)
    y_lvl4 = parity_xor

    # Nível 5: XORs + ANDs com pesos simbólicos
    padded = np.concatenate([X, X[:, :2]], axis=1)
    and_triplets = padded[:, :-2] & padded[:, 1:-1] & padded[:, 2:]

    xor_weights = 2 ** np.arange(1, xor_pairs.shape[1] + 1)
    and_weights = np.flip(np.arange(1, and_triplets.shape[1] + 1))

    xor_weighted_sum = np.sum(xor_pairs * xor_weights, axis=1)
    and_weighted_sum = np.sum(and_triplets * and_weights, axis=1)

    combined = (xor_weighted_sum + 3 * and_weighted_sum) % 2
    y_lvl5 = combined.reshape(-1, 1).astype(np.float32)

    # Nível 6: Hierarquia de XORs (redução tipo árvore)
    def hierarchical_xor(seq):
        while seq.shape[1] > 1:
            if seq.shape[1] % 2 != 0:
                seq = np.concatenate(
                    [seq, np.zeros((seq.shape[0], 1), dtype=int)], axis=1)
            seq = seq[:, ::2] ^ seq[:, 1::2]
        return seq

    y_lvl6 = hierarchical_xor(X).astype(np.float32)

    # Nível 7: Lógica Condicional Dinâmica
    def level7_dynamic_logic(X):
        outputs = []
        for seq in X:
            result = 0
            for i in range(0, len(seq) - 1, 2):
                control = seq[i]
                value = seq[i + 1]
                if control:
                    result ^= value
            outputs.append(result)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl7 = level7_dynamic_logic(X)

    # 🌪️ Nível 8: Meta-Routing Auto-Referente com Redução Invertida
    def level8_meta_routing(X):
        outputs = []
        for seq in X:
            route = []
            for i in range(0, len(seq) - 2, 3):
                a, b, c = seq[i], seq[i+1], seq[i+2]
                route.append((a & b) ^ c)
            if len(route) == 0:
                route_sum = 0
            else:
                route_sum = route[0]
                for bit in route[1:]:
                    route_sum ^= bit
            result = route_sum ^ (len(route) % 2)
            outputs.append(result)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl8 = level8_meta_routing(X)

    # ⚔️ Nível 9: Paridade de transições de estados acumulativos (máquina de estado simbólica)

    def level9_state_transitions(X):
        outputs = []
        for seq in X:
            state = 0
            for i in range(len(seq)):
                bit = seq[i]
                if i % 2 == 0:
                    state ^= bit  # transição por XOR
                else:
                    state = (state + bit) % 2  # transição por soma modular
            outputs.append(state)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl9 = level9_state_transitions(X)

    # 🚀 Nível 10: Meta-Paridade com Estado Injetado

    def level10_meta_parity(X):
        results = []
        for seq in X:
            # 1. Máscara simbólica: XOR entre pares invertidos
            mask = seq[::-1][::2] ^ seq[::2][:len(seq[::-1][::2])]
            # 2. Aplicar a máscara (estende se necessário)
            mask_full = np.resize(mask, len(seq))
            modified = seq ^ mask_full
            # 3. Estado interno: bit central (como "memória simbólica")
            state = modified[len(modified) // 2]
            # 4. Condicional: XOR se estado é 1, caso contrário AND entre blocos
            first_half = modified[:len(modified) // 2]
            second_half = modified[len(modified) // 2:]
            if state == 1:
                op = np.sum(first_half ^ second_half) % 2
            else:
                op = np.sum(first_half & second_half) % 2
            results.append(op)
        return np.array(results).reshape(-1, 1).astype(np.float32)

    y_lvl10 = level10_meta_parity(X)

    # 🌪️ Nível 11: Circuito Condicional Reentrante

    def level11_conditional_reentry(X):
        outputs = []
        for seq in X:
            acc = 0
            for i in range(len(seq) - 3):
                a, b, ctrl1, ctrl2 = seq[i], seq[i + 1], seq[i + 2], seq[i + 3]

                if ctrl1 and not ctrl2:
                    acc ^= a & b  # XOR entre ANDs se ctrl1 ativo
                elif ctrl2 and not ctrl1:
                    acc ^= a | b  # XOR entre ORs se ctrl2 ativo
                elif ctrl1 and ctrl2:
                    acc ^= ~(a ^ b) & 1  # XNOR limitado (bitwise)
                # se nenhum controle ativo, mantém estado

            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl11 = level11_conditional_reentry(X)

    # 🚀 Nível 12: Lógica Alternada Hierárquica
    def level12_alternating_logic(X):
        def apply_rule(a, b, rule_id):
            if rule_id == 0:
                return a ^ b        # XOR
            elif rule_id == 1:
                return a & b      # AND
            elif rule_id == 2:
                return a | b      # OR
            elif rule_id == 3:
                return ~(a & b) & 1  # NAND
            elif rule_id == 4:
                return ~(a ^ b) & 1  # XNOR
            else:
                return a ^ b  # fallback

        results = []
        for seq in X:
            result = seq[0]
            for i in range(1, len(seq)):
                rule_id = i % 5  # alterna entre 5 regras
                result = apply_rule(result, seq[i], rule_id)
            results.append(result)
        return np.array(results).reshape(-1, 1).astype(np.float32)

    y_lvl12 = level12_alternating_logic(X)

    def level13_recursive_program(X):
        outputs = []
        for seq in X:
            acc = 0
            for i in range(0, len(seq)-3, 3):
                ctrl1, ctrl2, val = seq[i], seq[i+1], seq[i+2]
                if ctrl1 and ctrl2:
                    acc ^= val  # XOR se dois controles ativos
                elif ctrl1:
                    acc |= val  # OR se só um controle ativo
                elif ctrl2:
                    acc &= val  # AND se só ctrl2 ativo
                else:
                    acc = ~acc & 1  # NOT se nenhum ativo
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl13 = level13_recursive_program(X)

    # ⚔️ Nível 14: Lógica Condicional Recursiva baseada em Cabeçalho Binário

    def level14_recursive_logic(X):
        outputs = []
        for seq in X:
            # Interpreta os 2 primeiros bits como modo de operação
            mode = (seq[0] << 1) | seq[1]  # 00, 01, 10, 11

            if mode == 0:
                result = np.sum(seq[2:6]) % 2
            elif mode == 1:
                result = (seq[2] ^ seq[3]) & (seq[4] | seq[5])
            elif mode == 2:
                result = int(((seq[2] ^ seq[3]) ^ (seq[6] & seq[7])) % 2 == 1)
            else:
                inner = seq[2:6] ^ seq[6:10]
                result = np.sum(inner) % 2

            outputs.append(result)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl14 = level14_recursive_logic(X)

    # 🧠 Nível 15: Paridade com múltiplos contextos lógicos cruzados
    def level15_context_switch(X):
        out = []
        for seq in X:
            res = 0
            flip = 0
            for i in range(0, len(seq) - 2, 3):
                a, b, c = seq[i:i+3]
                context = (a << 2) | (b << 1) | c
                if context in [0, 3, 5, 6]:  # contextos lógicos definidos
                    flip ^= (a & ~b) | (c & b)
                else:
                    flip ^= (a ^ c)
            out.append(flip)
        return np.array(out).reshape(-1, 1).astype(np.float32)

    y_lvl15 = level15_context_switch(X)

    # Nível 16: Paridade condicional por janelas com rota lógica de decisão

    def level16_windowed_conditional(X):
        outputs = []
        for seq in X:
            result = 0
            for i in range(len(seq) - 2):
                window = seq[i:i+3]
                if window[0] == 1 and window[2] == 1:
                    result ^= window[1]  # aplica XOR se bordas são 1
                elif window[0] == 0 and window[2] == 0:
                    # inverte e aplica XOR se bordas são 0
                    result ^= ~window[1] & 1
                else:
                    result ^= window[0] & window[2]  # aplica AND entre bordas
            outputs.append(result)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Adiciona ao retorno da função
    y_lvl16 = level16_windowed_conditional(X)

    # 🧠 Nível 17: Paridade com Roteamento Contextual Temporal
    def level17_temporal_context(X):
        outputs = []
        for seq in X:
            context = 0
            memory = 0
            for i in range(len(seq)):
                if i % 2 == 0:
                    context ^= seq[i]
                else:
                    if context:
                        memory += seq[i]
                    else:
                        memory -= seq[i]
            outputs.append(int(memory % 2 == 0))
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl17 = level17_temporal_context(X)

    # ⚔️ Nível 18: Multiplexação simbólica com controle hierárquico

    def level18_multiplexed_logic(X):
        results = []
        for seq in X:
            acc = 0
            for i in range(0, len(seq) - 3, 3):
                ctrl_1 = seq[i]
                ctrl_2 = seq[i + 1]
                val = seq[i + 2]

                # Multiplexador simbólico com controle hierárquico
                if ctrl_1 and not ctrl_2:
                    acc ^= val
                elif ctrl_2 and not ctrl_1:
                    acc &= val
                elif ctrl_1 and ctrl_2:
                    acc |= val
                # senão ignora
            results.append(acc)
        return np.array(results).reshape(-1, 1).astype(np.float32)

    y_lvl18 = level18_multiplexed_logic(X)

    def level19_meta_conditional(X, block_size=4):
        n_samples, seq_len = X.shape
        outputs = []
        for sample in X:
            result = 0
            for i in range(0, seq_len - block_size, block_size):
                control = sample[i]  # bit de controle do bloco
                data_block = sample[i+1:i+block_size]
                if control:
                    block_result = np.bitwise_and.reduce(data_block)
                else:
                    block_result = np.bitwise_xor.reduce(data_block)
                result ^= block_result  # combina o resultado final com XOR global
            outputs.append(result)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl19 = level19_meta_conditional(X)

    # ⚠️ Nível 20: Lógica simbólica cruzada com ciclos dinâmicos
    def level20_crossed_cycles(X):
        n = X.shape[1]
        out = []
        for seq in X:
            step1 = []
            acc = 0
            for i in range(n):
                acc += seq[i]
                logic = (seq[i] ^ (acc % 2)) if i % 3 == 0 else (
                    seq[i] & (acc % 2))
                step1.append(logic)
            step2 = []
            for i in range(n):
                prev = step1[i - 1] if i > 0 else step1[-1]
                next_ = step1[(i + 1) % n]
                val = (prev ^ next_) & step1[i]
                step2.append(val)
            final = sum(step2) % 2
            out.append(final)
        return np.array(out).reshape(-1, 1).astype(np.float32)

    y_lvl20 = level20_crossed_cycles(X)

    # ⚠️ Nível 21: Lógica Autocondicional com Espelhamento Pseudoaleatório

    def level21_recursive_logic(X):
        outputs = []
        for seq in X:
            seq = seq.tolist()
            acc = 0
            for i in range(0, len(seq) - 2):
                a = seq[i]
                b = seq[i + 1]
                c = seq[i + 2]

                # Regra condicional: se a é 1, aplica XOR(b, c), senão aplica XNOR(b, c)
                if a:
                    logic = b ^ c
                else:
                    logic = int(b == c)

                # Espelhamento condicional: se i par, inverte bit
                if i % 2 == 0:
                    logic = 1 - logic

                # Acumula com peso simbólico dinâmico (baseado em posição i)
                acc += logic * ((i + 3) % 5 + 1)

            outputs.append(acc % 2)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl21 = level21_recursive_logic(X)

    # 🔁 Nível 22: XOR entre subgrupos reversos com deslocamento dinâmico

    def level22_shifted_xor(X):
        seq_len = X.shape[1]
        shift = (np.sum(X, axis=1) % seq_len).astype(int)
        outputs = []
        for i, seq in enumerate(X):
            s = shift[i]
            rotated = np.roll(seq, s)
            left = rotated[:seq_len // 2]
            right = rotated[seq_len // 2:]
            xor_result = left ^ right[::-1]
            outputs.append(np.sum(xor_result) % 2)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl22 = level22_shifted_xor(X)

    # 🧠 Nível 23: XOR condicional com simulação de memória acumulativa

    def level23_memory_xor(X):
        outputs = []
        for seq in X:
            acc = 0
            for i, val in enumerate(seq):
                if i % 3 == 0:
                    acc ^= val
                elif i % 3 == 1:
                    acc ^= (acc & val)
                else:
                    acc ^= (~val & 1)
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl23 = level23_memory_xor(X)

    # 🤖 Nível 24: Simulação de um autômato binário (Regra 90-like)

    def level24_automaton(X):
        pad = np.pad(X, ((0, 0), (1, 1)), mode='constant')
        left = pad[:, :-2]
        right = pad[:, 2:]
        center = X
        next_gen = left ^ right
        y = np.sum(next_gen, axis=1) % 2
        return y.reshape(-1, 1).astype(np.float32)

    y_lvl24 = level24_automaton(X)

    # 📊 Nível 25: Cálculo de "peso lógico" com base em AND/OR

    def level25_logic_weight(X):
        logic_sum = np.sum((X[:, :-1] & X[:, 1:]) |
                           (~X[:, :-1] & ~X[:, 1:]), axis=1)
        y = (logic_sum % 2).astype(np.float32)
        return y.reshape(-1, 1)

    y_lvl25 = level25_logic_weight(X)

    # 🔀 Nível 26: XOR em grupos de 3 com reversões alternadas

    def level26_triplet_xor(X):
        outputs = []
        for seq in X:
            acc = 0
            for i in range(0, len(seq) - 2, 3):
                a, b, c = seq[i:i+3]
                group = [a, b, c][::-1] if i % 2 == 0 else [a, b, c]
                acc ^= group[0] ^ group[1] ^ group[2]
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl26 = level26_triplet_xor(X)

    # 🧩 Nível 27: Operação customizada baseada em padrões alternados

    def level27_pattern_logic(X):
        outputs = []
        for seq in X:
            acc = 1
            for i in range(1, len(seq)):
                if seq[i - 1] == 1 and seq[i] == 0:
                    acc ^= 1
                elif seq[i - 1] == 0 and seq[i] == 1:
                    acc &= 1
                else:
                    acc |= 0
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl27 = level27_pattern_logic(X)

    # 🧠 Nível 28: Condições aninhadas com XOR e OR simulando controle de fluxo

    def level28_nested_conditions(X):
        outputs = []
        for seq in X:
            acc = 0
            for i in range(len(seq)):
                if seq[i]:
                    acc ^= (seq[i - 1] if i > 0 else 0)
                else:
                    acc |= (seq[i - 2] if i > 1 else 0)
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl28 = level28_nested_conditions(X)

    # ⚙️ Nível 29: Combina lógica e posição (influência indexada)

    def level29_indexed_logic(X):
        outputs = []
        for seq in X:
            acc = 1
            for i, val in enumerate(seq):
                if i % 2 == 0:
                    acc ^= val
                else:
                    acc &= val
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl29 = level29_indexed_logic(X)

    # 🧠💥 Nível 30: Combinação final – lógica cruzada com simulação de meta-raciocínio

    def level30_meta_reasoning(X):
        xor_pairs = X[:, :-1:2] ^ X[:, 1::2]
        and_blocks = X[:, ::2] & X[:, 1::2]
        score = (np.sum(xor_pairs, axis=1) * 2 +
                 np.sum(and_blocks, axis=1)) % 2
        return score.reshape(-1, 1).astype(np.float32)

    y_lvl30 = level30_meta_reasoning(X)

    # 🧾 Nível 31: Máquina de Turing simbólica (leitura com regra de estado binário) AGI Simulada ~

    def level31_turing_machine(X):
        outputs = []
        for seq in X:
            state = 0
            head = 0
            tape = seq.copy()
            for _ in range(3):  # 3 "passos" na fita
                if tape[head] == 1:
                    state ^= 1
                    tape[head] = 0
                    head = (head + 1) % len(tape)
                else:
                    state |= 1
                    tape[head] = 1
                    head = (head - 1) % len(tape)
            outputs.append(state)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl31 = level31_turing_machine(X)

    # 🧠🔥 Nível 32: Máquina de Turing com Fita e Estado interno

    def level32_turing_machine(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())  # fita binária
            head = len(tape) // 2    # começa no meio da fita
            state = 0                # estado inicial

            for _ in range(10):  # número fixo de ciclos
                symbol = tape[head]

                if state == 0:
                    if symbol == 0:
                        tape[head] = 1
                        head = max(0, head - 1)
                        state = 1
                    else:
                        tape[head] = 0
                        head = min(len(tape) - 1, head + 1)
                        state = 2

                elif state == 1:
                    if symbol == 0:
                        head = min(len(tape) - 1, head + 1)
                        state = 2
                    else:
                        tape[head] = 1
                        head = max(0, head - 1)
                        state = 0

                elif state == 2:
                    if symbol == 1:
                        tape[head] = 0
                        state = 1
                    else:
                        head = max(0, head - 1)
                        state = 0

            output = sum(tape) % 2  # paridade final da fita após modificações
            outputs.append(output)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl32 = level32_turing_machine(X)

    # 🏗️ Nível 33: Simulação de máquina de Turing com empilhamento lógico reverso

    def level33_stack_sim(X):
        outputs = []
        for seq in X:
            stack = []
            for i, bit in enumerate(seq):
                if i % 2 == 0:
                    stack.append(bit)
                else:
                    if stack:
                        top = stack.pop()
                        bit = bit ^ top
                    else:
                        bit = bit
            result = bit if len(stack) % 2 == 0 else bit ^ 1
            outputs.append(result)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl33 = level33_stack_sim(X)

    # 🧠⚙️ Nível 34: Máquina de Turing simbólica minimalista (com estados e escrita)

    def level34_turing_machine(X):
        outputs = []

        for seq in X:
            tape = list(seq.copy())
            state = 0  # Começa no estado 0
            head = len(tape) // 2  # Começa no meio da fita

            # Define a tabela de transição: (estado_atual, valor_lido) -> (novo_valor, próximo_estado, movimento)
            transitions = {
                (0, 0): (1, 1, 1),
                (0, 1): (0, 0, -1),
                (1, 0): (1, 0, -1),
                (1, 1): (0, 1, 1),
            }

            steps = len(seq) * 2  # Número de passos de simulação
            for _ in range(steps):
                symbol = tape[head]
                key = (state, symbol)
                if key in transitions:
                    new_val, next_state, move = transitions[key]
                    tape[head] = new_val
                    state = next_state
                    head += move
                    # limita dentro da fita
                    head = max(0, min(len(tape) - 1, head))

            # Saída: paridade da fita modificada (simbolizando computação da máquina)
            result = sum(tape) % 2
            outputs.append(result)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl34 = level34_turing_machine(X)

    # 🧠🌀 Nível 35: Simulação de Máquina de Turing com estados, fita e controle condicional

    def level35_turing_sim(X):
        outputs = []
        for seq in X:
            state = 0
            head = 0
            tape = seq.copy().tolist()
            steps = len(tape) * 2

            for _ in range(steps):
                symbol = tape[head] if 0 <= head < len(tape) else 0

                # Transições de estado baseadas no símbolo
                if state == 0:
                    if symbol == 1:
                        state = 1
                        tape[head] = 0
                        head += 1
                    else:
                        head += 1
                elif state == 1:
                    if symbol == 0:
                        state = 2
                        tape[head] = 1
                        head -= 1
                    else:
                        head += 1
                elif state == 2:
                    if symbol == 1:
                        state = 0
                        head += 1
                    else:
                        tape[head] = 1
                        head -= 1

                if head < 0 or head >= len(tape):
                    break

            parity = sum(tape) % 2
            outputs.append(parity)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl35 = level35_turing_sim(X)

    # 🧠⚙️ Nível 36: Máquina de Turing com cabeça de leitura simbólica

    def level36_turing_machine(X):
        outputs = []
        for seq in X:
            state = 0
            for i in range(len(seq)):
                if state == 0:
                    state = 1 if seq[i] == 1 else 0
                elif state == 1:
                    state = 2 if seq[i] == 0 else 0
                elif state == 2:
                    state = 3 if seq[i] == 1 else 1
                elif state == 3:
                    state = 0 if seq[i] == 1 else 2
            # saída simbólica com base no estado final
            outputs.append(state % 2)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl36 = level36_turing_machine(X)

    # 🧵 Nível 37: Máquina de Turing binária com fita, transições e estado finito

    def level37_turing_machine(X):
        outputs = []
        for seq in X:
            state = 0
            head_pos = len(seq) // 2  # começa no meio
            tape = seq.copy()
            for _ in range(len(seq) * 2):  # número de passos
                symbol = tape[head_pos]
                # Transições com base no estado e símbolo lido
                if state == 0 and symbol == 0:
                    tape[head_pos] = 1
                    state = 1
                    head_pos += 1
                elif state == 0 and symbol == 1:
                    tape[head_pos] = 0
                    state = 0
                    head_pos -= 1
                elif state == 1 and symbol == 0:
                    tape[head_pos] = 1
                    state = 0
                    head_pos -= 1
                elif state == 1 and symbol == 1:
                    tape[head_pos] = 1
                    state = 1
                    head_pos += 1
                head_pos = max(0, min(head_pos, len(seq) - 1))  # limita a fita
            # saída final é o valor atual da cabeça
            outputs.append(tape[head_pos])
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl37 = level37_turing_machine(X)

    # 🧾 Nível 38: Simulação de máquina de Turing binária com uma única regra de transição

    def level38_turing(X):
        outputs = []
        for seq in X:
            state = 0
            head = len(seq) // 2
            tape = list(seq.copy())
            for _ in range(10):  # até 10 passos de execução
                read = tape[head]
                if state == 0 and read == 1:
                    tape[head] = 0
                    head = max(0, head - 1)
                    state = 1
                elif state == 1 and read == 0:
                    tape[head] = 1
                    head = min(len(seq) - 1, head + 1)
                    state = 0
                else:
                    break
            outputs.append(tape[head])
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl38 = level38_turing(X)

    # 🧠⚙️ Nível 39: Máquina de Turing Simbólica com Cabeçote Deslizante = AGI Lite

    def level39_turing_machine(X):
        outputs = []
        for seq in X:
            head = 0
            tape = seq.copy()
            acc = 0

            for _ in range(len(seq)):
                curr = tape[head]

                if curr == 1:
                    acc ^= 1
                    head = (head + 2) % len(seq)
                else:
                    acc &= 1
                    head = (head - 1) % len(seq)

            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl39 = level39_turing_machine(X)

    # 🧠🔁 Nível 40: Máquina de Turing Lógica com Cabeça Simbólica

    def level40_turing_head(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            head = 0
            state = 1

            for _ in range(len(tape)):
                symbol = tape[head]
                if state == 1:
                    if symbol == 0:
                        tape[head] = 1
                        head = (head + 1) % len(tape)
                        state = 2
                    else:
                        tape[head] = 0
                        head = (head - 1) % len(tape)
                        state = 3
                elif state == 2:
                    tape[head] ^= 1
                    head = (head + 2) % len(tape)
                    state = 1
                elif state == 3:
                    tape[head] |= 1
                    head = (head + 1) % len(tape)
                    state = 2

            # A saída é a soma da fita final, paridade da soma e posição final da cabeça
            acc = (sum(tape) + head) % 2
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl40 = level40_turing_head(X)

    # ⚙️🧠 Nível 41 - Dupla Cabeça de Turing com Feedback Cruzado

    def level41_double_heads(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            h1, h2 = 0, len(seq) // 2
            s1, s2 = 1, 1

            for _ in range(len(seq)):
                v1, v2 = tape[h1], tape[h2]

                if s1 == 1:
                    tape[h1] ^= v2
                    h1 = (h1 + 1) % len(seq)
                    s1 = 2
                else:
                    tape[h1] |= v1
                    h1 = (h1 - 1) % len(seq)
                    s1 = 1

                if s2 == 1:
                    tape[h2] &= v1
                    h2 = (h2 + 2) % len(seq)
                    s2 = 2
                else:
                    tape[h2] ^= 1
                    h2 = (h2 - 2) % len(seq)
                    s2 = 1

            acc = (sum(tape) + h1 + h2 + s1 + s2) % 2
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl41 = level41_double_heads(X)

    # 🧠👑 Nível 42 - Máquina de Turing Simbólica com 3 Cabeças, Estados Dinâmicos e Simetria Espelhada

    def level42_triple_heads_symmetry(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            h1, h2, h3 = 0, n // 3, 2 * n // 3
            s1, s2, s3 = 0, 1, 2

            for _ in range(n):
                v1, v2, v3 = tape[h1], tape[h2], tape[h3]

                # Atualiza fita com lógica cruzada
                if s1 == 0:
                    tape[h1] ^= v2 & v3
                    h1 = (h1 + 1) % n
                    s1 = 1
                else:
                    tape[h1] = (v1 | v2) ^ v3
                    h1 = (h1 - 1) % n
                    s1 = 0

                if s2 == 1:
                    tape[h2] ^= v1
                    h2 = (h2 + 2) % n
                    s2 = 2
                else:
                    tape[h2] &= v3
                    h2 = (h2 - 2) % n
                    s2 = 1

                if s3 == 2:
                    tape[h3] = (~v2) & 1
                    h3 = (h3 + 3) % n
                    s3 = 0
                else:
                    tape[h3] |= v1 ^ v2
                    h3 = (h3 - 1) % n
                    s3 = 2

            # Espelhamento da fita e cálculo de padrão simétrico
            mirrored = tape[::-1]
            pattern_score = sum(1 for a, b in zip(tape, mirrored) if a == b)

            acc = (pattern_score + h1 + h2 + h3 + s1 + s2 + s3) % 2
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl42 = level42_triple_heads_symmetry(X)

    # 🧠🌀 Nível 43 – Máquina de Turing Simbólica com Reflexão Temporal, Rotação Circular e Paridade Cruzada

    def level43_time_reflection_parity(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            h1, h2 = 0, n // 2
            s1, s2 = 0, 1
            history = []

            for step in range(n):
                v1, v2 = tape[h1], tape[h2]

                # Históricos das posições e estados
                history.append((h1, h2, s1, s2))

                # Cabeça 1 – lógica reflexiva condicional
                if s1 == 0:
                    tape[h1] = v1 ^ v2
                    h1 = (h1 + 1) % n
                    s1 = 1
                else:
                    tape[h1] = v1 & (~v2 & 1)
                    h1 = (h1 - 1) % n
                    s1 = 0

                # Cabeça 2 – lógica cruzada + rotação
                if s2 == 1:
                    tape[h2] = (v2 | v1) ^ 1
                    h2 = (h2 + 2) % n
                    s2 = 2
                else:
                    tape[h2] ^= (h1 % 2)
                    h2 = (h2 - 2) % n
                    s2 = 1

            # 🔁 Reflexão Temporal: volta parcial no histórico
            for h1, h2, s1, s2 in reversed(history[:n//2]):
                tape[h1] ^= 1
                tape[h2] |= s1 ^ s2

            # 🔄 Rotação circular final
            rotated = tape[n//2:] + tape[:n//2]

            # 🧮 Paridade cruzada entre primeira e segunda metade
            half1, half2 = rotated[:n//2], rotated[n//2:]
            acc = sum([a ^ b for a, b in zip(half1, half2)]) % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl43 = level43_time_reflection_parity(X)

    # 🧠💾 Nível 44 – Máquina Simbólica Auto-Transformadora com Escrita de Regras Dinâmicas

    def level44_self_modifying_machine(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            rules = {}  # regras de transição dinâmicas
            head = 0
            state = 1

            for _ in range(n):
                symbol = tape[head]
                key = (state, symbol)

                # Cria uma nova regra se ainda não existir
                if key not in rules:
                    rules[key] = (1 - symbol, (head + symbol + state) %
                                  n, (state + symbol + 1) % 4)

                write_val, move_to, new_state = rules[key]

                tape[head] = write_val
                head = move_to
                state = new_state

                # Auto-transformação simbólica de regras (meta-escrita)
                if len(rules) > 3:
                    oldest_key = list(rules.keys())[0]
                    rules[oldest_key] = (
                        rules[oldest_key][0] ^ 1,
                        (rules[oldest_key][1] + 1) % n,
                        (rules[oldest_key][2] + 1) % 5,
                    )

            checksum = (sum(tape) + head + state + len(rules)) % 2
            outputs.append(checksum)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl44 = level44_self_modifying_machine(X)

    # ⚙️🧠 Nível 45 - Meta-Turing com Contexto Persistente

    def level45_meta_turing_context(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Três cabeças, cada uma com seu próprio estado
            h1, h2, h3 = 0, n // 3, 2 * n // 3
            s1, s2, s3 = 0, 1, 2

            # Memória simbólica persistente entre iterações
            context = 0

            for _ in range(n):
                v1, v2, v3 = tape[h1], tape[h2], tape[h3]

                # Lógica cruzada com mutações contextuais
                if s1 == 0:
                    tape[h1] = v1 ^ (v2 & context)
                    context ^= v3
                    h1 = (h1 + 1) % n
                    s1 = 1
                else:
                    tape[h1] = (~(v1 | context)) & 1
                    h1 = (h1 - 1) % n
                    s1 = 0

                if s2 == 1:
                    tape[h2] ^= (v1 | v3)
                    context |= v2
                    h2 = (h2 + 2) % n
                    s2 = 2
                else:
                    tape[h2] = (v2 & context) ^ v1
                    h2 = (h2 - 2) % n
                    s2 = 1

                if s3 == 2:
                    tape[h3] = (v3 ^ context) & 1
                    context ^= (v1 & v2)
                    h3 = (h3 + 3) % n
                    s3 = 0
                else:
                    tape[h3] = (v1 | v2 | v3 | context) % 2
                    h3 = (h3 - 1) % n
                    s3 = 2

            # Final: paridade do padrão mais score da simetria da fita
            mirrored = tape[::-1]
            sym_score = sum(1 for a, b in zip(tape, mirrored) if a == b)
            acc = (sum(tape) + h1 + h2 + h3 + context + sym_score) % 2

            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl45 = level45_meta_turing_context(X)

    # Nível 46 – “Máquina de Estados Hierárquicos com Loop de Consciência” 🧠🌀

    def level46_conscious_hierarchy(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Inicialização de três cabeças independentes
            h1, h2, h3 = 0, n // 3, 2 * n // 3
            s1, s2, s3 = 1, 2, 3  # Estados simbólicos iniciais
            last_states = [1, 2, 3]  # Memória reflexiva de estados anteriores

            for _ in range(n):

                v1, v2, v3 = tape[h1], tape[h2], tape[h3]
                ref = last_states[_ % 3]  # estado anterior mais distante

                # Camada Lógica 1 - processamento direto
                if s1 % 2 == 1:
                    tape[h1] ^= (v2 | v3)
                    h1 = (h1 + 1) % n
                    s1 = (s1 + ref) % 4
                else:
                    tape[h1] = (v1 & ~v2) | v3
                    h1 = (h1 - 1) % n
                    s1 = (s1 + 2) % 5

                # Camada Lógica 2 - cruzamento dinâmico
                if s2 in [2, 3]:
                    tape[h2] ^= (v1 & v3)
                    h2 = (h2 + 2) % n
                    s2 = (s2 + ref) % 4
                else:
                    tape[h2] |= (v2 ^ v1)
                    h2 = (h2 - 2) % n
                    s2 = (s2 + 1) % 5

                # Camada Lógica 3 - metaaprendizado (autoajuste simbólico)
                if s3 == 3:
                    tape[h3] = (~v1 & v2) ^ ref
                    h3 = (h3 + 3) % n
                    s3 = (s3 + v3 + 1) % 4
                else:
                    tape[h3] |= (v3 ^ ref)
                    h3 = (h3 - 1) % n
                    s3 = (s3 + v2) % 5

                # Atualiza memória reflexiva
                last_states = [s1, s2, s3]

            # Módulo final: loop de consciência (espelho do início + estado)
            mirrored = tape[::-1]
            sim = sum(1 for a, b in zip(tape, mirrored) if a == b)

            # Saída simbólica composta
            acc = (sim + sum(tape) + s1 + s2 + s3 + h1 + h2 + h3) % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl46 = level46_conscious_hierarchy(X)

    # 🧠🕰️ Nível 47 - Consciência Temporal Bidirecional com Consistência Histórica

    def level47_conscious_turing(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            history = []
            forward_head, backward_head = 0, n - 1
            state = 0
            consistent = True

            for t in range(n // 2):
                # Forward step
                symbol_fwd = tape[forward_head]
                history.append((forward_head, symbol_fwd, state))
                state = (state + symbol_fwd + forward_head) % 4
                tape[forward_head] ^= (state % 2)
                forward_head = (forward_head + 1) % n

                # Backward step (simultâneo)
                symbol_bwd = tape[backward_head]
                expected_state = (state + symbol_bwd + backward_head) % 4
                if abs(expected_state - state) > 1:
                    consistent = False
                state = expected_state
                tape[backward_head] ^= (state % 2)
                backward_head = (backward_head - 1) % n

            # Quantização de estados finais
            final_state = (sum(tape) + state +
                           forward_head + backward_head) % 8
            acc = int(consistent and (final_state % 2 == 0))
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Para gerar o y:
    y_lvl47 = level47_conscious_turing(X)

    # 🧠🌀 Nível 48 — Espelho Recursivo com Interferência de Estados

    def level48_recursive_mirror_interference(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            memory = [0] * n
            influence = 0

            for i in range(n):
                mirror_idx = n - i - 1
                if tape[i] == tape[mirror_idx]:
                    influence += 1
                    memory[i] = 1
                else:
                    memory[i] = memory[i - 1] if i > 0 else 0

                # Interferência simbólica: padrões anteriores mudam a interpretação atual
                if i % 3 == 0 and i > 1:
                    memory[i] ^= memory[i - 2]

            # A saída depende da soma da memória simbólica e do padrão de interferência
            acc = (sum(memory) + influence + tape[-1]) % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl48 = level48_recursive_mirror_interference(X)

    # 🔄🧠 Nível 49: Paridade Temporal com Consistência Reversa e Estados Históricos Paralelos

    def level49_temporal_reversal(X):
        outputs = []
        for seq in X:
            forward_state = 0
            reverse_state = 1
            mirror_seq = seq[::-1]
            history_forward = []
            history_reverse = []

            # Caminho normal (esquerda → direita)
            for i, bit in enumerate(seq):
                forward_state ^= (bit + i) % 2
                history_forward.append(forward_state)

            # Caminho reverso (direita → esquerda)
            for i, bit in enumerate(mirror_seq):
                reverse_state ^= (bit * (i % 3)) % 2
                history_reverse.append(reverse_state)

            # Combinação simbólica cruzada entre os dois históricos
            combined = []
            for a, b in zip(history_forward, history_reverse[::-1]):
                combined.append((a ^ b) & 1)

            # A saída depende da paridade da soma dos estados cruzados e consistência do início e fim
            parity = sum(combined) % 2
            consistent = int(seq[0] == seq[-1])
            acc = (parity + consistent) % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Uso:
    y_lvl49 = level49_temporal_reversal(X)

    # 🧠🪞 Nível 50 — Máquina de Autorreflexão Simbólica com Bifurcação de Consciência

    def level50_symbolic_self_reflection(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Dois "eus" simbólicos com cabeças, estados e memórias independentes
            heads = [0, n // 2]
            states = [1, 1]
            memories = [[0] * n, [0] * n]

            for _ in range(n):
                for i in range(2):  # cada "eu"
                    h = heads[i]
                    s = states[i]
                    v = tape[h]

                    # Simula transformação simbólica do estado interno
                    if s == 1:
                        tape[h] ^= 1
                        memories[i][h] = (memories[i][h] + 1) % 2
                        heads[i] = (h + 1) % n
                        states[i] = 2
                    else:
                        tape[h] ^= memories[i][h]
                        memories[i][h] = (memories[i][h] + tape[h]) % 2
                        heads[i] = (h - 1) % n
                        states[i] = 1

            # Reflexão cruzada: comparam inconsistências entre os dois mundos
            diffs = sum(1 for a, b in zip(memories[0], memories[1]) if a != b)
            consistency = sum(1 for a, b in zip(tape, tape[::-1]) if a == b)

            # Saída codifica reflexo simbólico entre os dois eus e o universo
            acc = (diffs + consistency + sum(heads) + sum(states)) % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl50 = level50_symbolic_self_reflection(X)

    # 🧠🔮 Nível 51 – Máquina de Simulação de Teoria da Mente com Projeção Contrafactual

    def level51_theory_of_mind(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Três cabeças simulam "mentes"
            h1, h2, h3 = 0, n // 3, 2 * n // 3
            s1, s2, s3 = 1, 1, 1

            for _ in range(n):
                v1, v2, v3 = tape[h1], tape[h2], tape[h3]

                # Cabeça 1 simula o que a 2 pensaria da 3 no passado
                mind2_about3 = (v3 ^ s2) & 1
                h1 = (h1 + mind2_about3 + 1) % n
                s1 = (s1 + mind2_about3) % 3

                # Cabeça 2 simula o que a 3 pensa da 1
                mind3_about1 = ((v1 | s3) ^ 1) & 1
                h2 = (h2 + mind3_about1 + 2) % n
                s2 = (s2 + mind3_about1) % 3

                # Cabeça 3 simula o que a 1 pensa da 2, e projeta contrafactualmente (inverso do que seria)
                mind1_about2 = (~(v2 ^ s1)) & 1
                h3 = (h3 - mind1_about2 - 1) % n
                s3 = (s3 + mind1_about2) % 3

                # Atualiza a fita com a intersecção das percepções simuladas
                tape[h1] ^= mind2_about3
                tape[h2] ^= mind3_about1
                tape[h3] ^= mind1_about2

            # A saída depende do padrão de consistência entre as projeções mentais cruzadas
            projection_score = (s1 + s2 + s3 + h1 + h2 + h3 + sum(tape)) % 2
            outputs.append(projection_score)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Aplicação
    y_lvl51 = level51_theory_of_mind(X)

    # 🧠🌌 Nível 52 - Máquina Quântica de Estados Paralelos com Múltiplas Realidades

    def level52_quantum_realities(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            heads = [0, len(seq)//3, 2*len(seq)//3]
            states = [1, 2, 3]
            histories = [[], [], []]

            for _ in range(len(tape)):
                for i in range(3):
                    pos = heads[i]
                    val = tape[pos]
                    state = states[i]

                    # Simulação de realidades paralelas
                    if state == 1:
                        tape[pos] ^= 1
                        heads[i] = (pos + 1) % len(tape)
                        states[i] = 2
                    elif state == 2:
                        tape[pos] = tape[pos] ^ ((val << 1) | 1) % 2
                        heads[i] = (pos + 2) % len(tape)
                        states[i] = 3
                    elif state == 3:
                        tape[pos] = (tape[pos] + sum(tape) + i) % 2
                        heads[i] = (pos - 1) % len(tape)
                        states[i] = 1

                    histories[i].append(tape[pos])

            # A saída depende da coerência entre realidades (históricos convergentes)
            agreement = sum(1 for a, b, c in zip(*histories) if a == b == c)
            acc = (agreement + sum(heads) + sum(states)) % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl52 = level52_quantum_realities(X)

    # 🌀🧠 Nível 53 - Máquina Temporal com Causalidade Cruzada e Espelhos Históricos

    def level53_temporal_causal_mirror(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            mirror_tape = tape[::-1]
            acc = 0

            for i in range(n):
                t = tape[i]
                m = mirror_tape[i]
                forward = tape[(i+1) % n]
                backward = tape[(i-1) % n]

                # Causalidade temporal cruzada
                logic = (t & forward) ^ (m | backward)
                logic = (logic + acc) % 2

                # Espelho histórico (simetria com peso decrescente)
                weight = 1.0 - abs(i - n//2) / (n//2)
                acc += int(logic * weight)

            # Paridade da soma acumulada simbólica
            out = acc % 2
            outputs.append(out)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl53 = level53_temporal_causal_mirror(X)

    # 🔥🧠 Nível 54 – Máquina de Consciência Temporal Bidirecional com Consistência Histórica e Estados Quantizados

    def level54_symbolic_consciousness(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Cabeças bidirecionais com histórico simbólico cruzado
            h1, h2 = 0, n - 1
            s1, s2 = 1, 2
            history = []

            for t in range(n):
                v1, v2 = tape[h1], tape[h2]

                # Estado s1 com lógica ilusória (operações reversíveis)
                if s1 == 1:
                    tape[h1] ^= v2
                    s1 = 2
                else:
                    tape[h1] = (~v1) & 1
                    s1 = 1
                h1 = (h1 + 1) % n

                # Estado s2 com mutações cíclicas e resgate histórico
                if s2 == 2:
                    tape[h2] |= v1
                    s2 = 3
                elif s2 == 3:
                    tape[h2] ^= v2
                    s2 = 1
                h2 = (h2 - 1) % n

                # Armazena a simetria momentânea
                pattern = int(v1 == v2)
                history.append(pattern)

            # Consistência histórica simbólica cruzada
            symmetry_score = sum(history[i] == history[-1 - i]
                                 for i in range(n // 2))
            state_signature = (s1 * 3 + s2) % 5
            final_sum = sum(tape) + symmetry_score + state_signature

            acc = final_sum % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl54 = level54_symbolic_consciousness(X)

    # 🧠⏳ Nível 55 – Meta-temporal com ruído adaptativo e eventos ocultos

    def level55_meta_temporal_hidden_noise(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            hidden_state = 1
            # ruído adaptativo baseado na própria entrada
            noise_gate = (sum(tape[:3]) % 2)
            ghost_memory = [0] * n  # memória fantasma invisível

            for i in range(n):
                symbol = tape[i]
                past = tape[i - 1] if i > 0 else 0
                future = tape[i + 1] if i < n - 1 else 0

                # Evento oculto simbólico
                if (i + hidden_state) % 3 == 0:
                    ghost_memory[i] = 1
                    hidden_state = (hidden_state + symbol + past + future) % 4
                else:
                    ghost_memory[i] = 0
                    hidden_state = (hidden_state + 1) % 4

                # Ruído adaptativo: muda bits irrelevantes, mas que confundem
                if (i % 2 == noise_gate):
                    tape[i] ^= (past & ~future) & 1

            # Resultado depende da consistência entre a fita e a memória oculta
            aligned = sum(1 for a, b in zip(tape, ghost_memory) if a == b)
            acc = (aligned + hidden_state + noise_gate + sum(ghost_memory)) % 2
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl55 = level55_meta_temporal_hidden_noise(X)

    # Nível 56 – Metaespelhamento Causal Recursivo

    def level56_meta_mirror_causal(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Três cabeças com deslocamentos espelhados
            h1, h2, h3 = 0, n // 2, n - 1
            s1, s2, s3 = 0, 1, 2

            history = []

            for _ in range(n):
                v1, v2, v3 = tape[h1], tape[h2], tape[h3]

                # Regras causais com simetria condicional
                if s1 == 0:
                    tape[h1] ^= v2 & (~v3 & 1)
                    s1 = 1
                else:
                    tape[h1] = (v1 | v3) ^ v2
                    s1 = 0
                h1 = (h1 + 1) % n

                if s2 == 1:
                    tape[h2] = (v2 & v1) ^ (~v3 & 1)
                    s2 = 2
                else:
                    tape[h2] ^= (v3 | v1)
                    s2 = 1
                h2 = (h2 - 1) % n

                if s3 == 2:
                    tape[h3] = v1 ^ v2 ^ v3
                    s3 = 0
                else:
                    tape[h3] = (v1 & v2) | (~v3 & 1)
                    s3 = 2
                h3 = (h3 + 2) % n

                # Armazena histórico para simular reversão posterior
                history.append(tuple(tape))

            # Metaespelhamento: compara reversão simulada do histórico
            reversed_check = True
            for i in range(n // 2):
                left = history[i]
                right = history[-i-1][::-1]
                if left != right:
                    reversed_check = False
                    break

            acc = 1 if reversed_check else 0
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl56 = level56_meta_mirror_causal(X)

    # Nivel 57 - MetaLógica Temporal: paridade dos 4 primeiros ⊕ paridade cruzada dos 4 últimos

    def level57_meta_causal_dupla_paridade(X):
        outputs = []
        for seq in X:
            seq = seq.astype(int)
            n = len(seq)
            metade = n // 2

            A = list(seq[:metade])
            B = list(seq[metade:])

            # Cabeças de leitura para A e B
            ha, hb = 0, len(B) - 1
            sa, sb = 0, 0

            history_a, history_b = [], []

            for _ in range(metade):
                va = A[ha]
                vb = B[hb]

                # Regras causais diferentes para cada metade
                if sa == 0:
                    A[ha] ^= (va & 1)
                    sa = 1
                else:
                    A[ha] = (va ^ ha) % 2
                    sa = 0
                ha = (ha + 1) % metade

                if sb == 0:
                    B[hb] = (vb | (hb % 2)) ^ 1
                    sb = 1
                else:
                    B[hb] ^= (vb & (hb % 2))
                    sb = 0
                hb = (hb - 1) % metade

                history_a.append(A.copy())
                history_b.append(B.copy())

            # Paridade simples de A
            par_A = sum(A) % 2

            # Paridade cruzada de B (posição par ⊕ posição ímpar)
            cruzada = 0
            for i in range(0, len(B)-1, 2):
                cruzada ^= B[i] ^ B[i+1]

            acc = par_A ^ cruzada
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Exemplo de uso:
    y_lvl57 = level57_meta_causal_dupla_paridade(X)

    # Consciência Simbólica Temporal com Memória e Simetria Cruzada.

    def level58_consciencia_simbolica_temporal(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            heads = [0, n//3, 2*n//3]
            states = [0, 1, 2]
            mem = []

            consistent_history = True

            for step in range(n):
                vals = [tape[h] for h in heads]
                combined = sum(vals) % 2

                # Regras cruzadas com memória interna e consistência temporal
                if states[0] == 0:
                    tape[heads[0]] ^= vals[1]
                    states[0] = 1
                else:
                    tape[heads[0]] = vals[2] ^ combined
                    states[0] = 0
                heads[0] = (heads[0] + 1) % n

                if states[1] == 1:
                    tape[heads[1]] = (vals[1] & vals[0]) | (~vals[2] & 1)
                    states[1] = 2
                else:
                    tape[heads[1]] ^= vals[0]
                    states[1] = 1
                heads[1] = (heads[1] - 1) % n

                if states[2] == 2:
                    tape[heads[2]] = vals[0] ^ vals[1] ^ vals[2]
                    states[2] = 0
                else:
                    tape[heads[2]] = (vals[2] & vals[1]) | (~vals[0] & 1)
                    states[2] = 2
                heads[2] = (heads[2] + 2) % n

                mem.append(tuple(tape))

            # Checagem de consistência entre padrões simbólicos históricos
            for i in range(1, len(mem)//3):
                past = mem[i]
                future = mem[-i-1][::-1]
                if past != future:
                    consistent_history = False
                    break

            acc = 1 if consistent_history else 0
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl58 = level58_consciencia_simbolica_temporal(X)

    def level59_meta_inverse_conditional_transitivity(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Cabeças que se deslocam de forma reversível e condicional
            h1, h2, h3 = 0, n // 3, 2 * n // 3
            s1, s2, s3 = 1, 0, 2

            transitions = []

            for _ in range(n):
                v1, v2, v3 = tape[h1], tape[h2], tape[h3]

                # Regras com inversão condicional e simetria transitiva
                if s1 == 1:
                    tape[h1] = v2 ^ v3
                    s1 = 0
                else:
                    tape[h1] = (v1 & ~v2 & 1) | v3
                    s1 = 1
                h1 = (h1 + 1) % n

                if s2 == 0:
                    tape[h2] = (v2 | v3) ^ v1
                    s2 = 2
                else:
                    tape[h2] = (~v1 & 1) ^ (v2 & v3)
                    s2 = 0
                h2 = (h2 - 1) % n

                if s3 == 2:
                    tape[h3] = v1 ^ v2 ^ v3
                    s3 = 1
                else:
                    tape[h3] = (v1 & v2) | (v3 ^ 1)
                    s3 = 2
                h3 = (h3 + 2) % n

                # Armazena a transformação para testar transitividade reversa
                transitions.append(tuple(tape))

            # Verificação: cada transição deve ser reversível por pares espelhados
            reversible = True
            for i in range(1, len(transitions) // 2):
                if transitions[i] != transitions[-i - 1][::-1]:
                    reversible = False
                    break

            acc = 1 if reversible else 0
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl59 = level59_meta_inverse_conditional_transitivity(X)

    def level60_mirror_machine_perturb(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Cabeças espelhadas
            h1, h2 = 0, n - 1
            state = 0
            perturb = 1

            history = []

            for _ in range(n):
                v1, v2 = tape[h1], tape[h2]

                if state == 0:
                    tape[h1] ^= (v2 ^ perturb)
                    tape[h2] ^= (v1 & perturb)
                    state = 1
                else:
                    tape[h1] = (v1 | perturb) ^ v2
                    tape[h2] = (v2 & ~perturb & 1) | v1
                    state = 0

                history.append(tuple(tape))
                h1 = (h1 + 1) % n
                h2 = (h2 - 1) % n

                # alterna perturbação de forma controlada
                perturb = (perturb + 1) % 2

            # Condição: reverso espelhado com tolerância simétrica
            passed = True
            for i in range(n // 2):
                left = history[i]
                right = history[-i - 1][::-1]
                diffs = sum(l != r for l, r in zip(left, right))
                if diffs > 1:  # tolerância simbólica
                    passed = False
                    break

            acc = 1 if passed else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl60 = level60_mirror_machine_perturb(X)

    def level61_parallel_temporal_clocks(X):
        outputs = []
        for seq in X:
            n = len(seq)
            cycle_lengths = [4, 6, 9]
            clocks = [0, 0, 0]
            positions = [0, 0, 0]
            tapes = [[0]*length for length in cycle_lengths]

            for t, bit in enumerate(seq):
                for i in range(3):
                    pos = positions[i] % cycle_lengths[i]
                    if bit == 1:
                        tapes[i][pos] ^= 1
                    else:
                        tapes[i][pos] = (tapes[i][pos] + 1) % 2
                    positions[i] += 1

            final_states = [sum(tape) % 2 for tape in tapes]
            output = 1.0 if final_states.count(final_states[0]) == 3 else 0.0
            outputs.append(output)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl61 = level61_parallel_temporal_clocks(X)

    def level62_temporal_scramble_restore(X):
        outputs = []

        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Etapa 1: Embaralhamento temporal condicional reversível
            scrambled = tape.copy()
            for i in range(1, n - 1, 2):
                scrambled[i], scrambled[i + 1] = scrambled[i + 1], scrambled[i]

            # Etapa 2: Reconstrução reversa condicional com regra de paridade simétrica
            reconstructed = scrambled.copy()
            for i in range(n):
                if i % 3 == 0:
                    reconstructed[i] = scrambled[i] ^ scrambled[(i + 1) % n]
                elif i % 3 == 1:
                    reconstructed[i] = scrambled[i - 1] ^ scrambled[i]
                else:
                    reconstructed[i] = scrambled[i] ^ scrambled[(i - 2) % n]

            # Etapa 3: Verificação – o modelo deve inferir se a reconstrução equivale ao original
            is_correct = int(reconstructed == tape)
            outputs.append(is_correct)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl62 = level62_temporal_scramble_restore(X)

    # 🧠 Nível 63 – Consistência Temporal Contraditória com Meta-Reversão

    def level63_contradictory_temporal_logic(X):
        outputs = []

        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Três cabeças com estados
            h1, h2, h3 = 0, n // 3, (2 * n) // 3
            s1, s2, s3 = 0, 1, 2

            # Histórico de reversões simuladas
            history = []

            for i in range(n):
                v1, v2, v3 = tape[h1], tape[h2], tape[h3]

                # Reversão simulada com contradição lógica:
                if s1 == 0:
                    tape[h1] = v2 ^ (~v3 & 1)
                    s1 = 1
                else:
                    tape[h1] = (v1 | v3) & v2
                    s1 = 0
                h1 = (h1 + 1) % n

                if s2 == 1:
                    tape[h2] ^= (v1 & v3)
                    s2 = 2
                else:
                    tape[h2] = (~v2 & 1) ^ v3
                    s2 = 1
                h2 = (h2 + 2) % n

                if s3 == 2:
                    tape[h3] = (v1 ^ v2 ^ v3)
                    s3 = 0
                else:
                    tape[h3] = ((v1 & v2) | (~v3 & 1))
                    s3 = 2
                h3 = (h3 - 1) % n

                history.append(tuple(tape))

            # Verificação de consistência reversa lógica:
            consistent = True
            for i in range(n // 2):
                left = history[i]
                right = history[-i-1]
                contradiction = sum(a ^ b for a, b in zip(
                    left, right[::-1])) > n // 4
                if contradiction:
                    consistent = False
                    break

            acc = 1 if consistent else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl63 = level63_contradictory_temporal_logic(X)

    def level64_causal_mirror_loop(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Cabeças com deslocamentos cruzados
            h1, h2, h3 = 0, n//3, 2*n//3
            s1, s2, s3 = 0, 1, 2
            loop_check = []

            for i in range(n):
                v1, v2, v3 = tape[h1], tape[h2], tape[h3]

                # Atualizações com reversão simbólica alternada
                if s1 == 0:
                    tape[h1] = (v2 ^ v3) & (~v1 & 1)
                    s1 = 1
                else:
                    tape[h1] ^= (v2 | v1)
                    s1 = 0
                h1 = (h1 + 1) % n

                if s2 == 1:
                    tape[h2] = (~v3 & 1) ^ (v1 & v2)
                    s2 = 2
                else:
                    tape[h2] = v1 ^ v2 ^ v3
                    s2 = 1
                h2 = (h2 + 2) % n

                if s3 == 2:
                    tape[h3] ^= v1 | (~v2 & 1)
                    s3 = 0
                else:
                    tape[h3] = (v3 & v1) ^ v2
                    s3 = 2
                h3 = (h3 - 1) % n

                # Salva estado parcial para verificar se o loop volta ao espelho
                loop_check.append(tuple(tape))

            # Correção de loop: a primeira metade deve ser espelho da segunda
            valid = True
            for i in range(n // 2):
                if loop_check[i] != loop_check[-i-1][::-1]:
                    valid = False
                    break

            acc = 1 if valid else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl64 = level64_causal_mirror_loop(X)

    def level65_triptych_interpreter(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Divide em três partes: início, meio e fim
            third = n // 3
            start = tape[:third]
            middle = tape[third:2*third]
            end = tape[2*third:]

            # Fases com significados simbólicos:
            # Start: intenção, Middle: conflito, End: resolução

            # Interpreta a intenção
            intent = sum(start) % 2  # 0: estável, 1: desejando mudança

            # Interpreta o conflito como soma dos XORs dos pares do meio
            conflict = 0
            for i in range(0, len(middle)-1, 2):
                conflict ^= middle[i] ^ middle[i+1]

            # Interpreta a resolução como um delta de consistência
            resolution = 1 if sum(end) % 2 == conflict else 0

            # A lógica da máquina: se intenção e resolução batem, aceita
            result = 1 if resolution == intent else 0

            outputs.append(result)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl65 = level65_triptych_interpreter(X)

    y_lvl66 = None

    def level67_meta_cognition_symbolic(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            internal_state = [0] * n
            audit_log = []

            # Cabeças simbólicas que escrevem julgamentos internos
            h1, h2 = 0, n // 2
            s1, s2 = 0, 1

            for _ in range(n):
                v1, v2 = tape[h1], tape[h2]

                # Regra simbólica que aplica inferência e memorização interna
                if s1 % 2 == 0:
                    inferred = (v1 ^ v2) & 1
                else:
                    inferred = (~(v1 & v2)) & 1
                internal_state[h1] = inferred
                s1 += 1
                h1 = (h1 + 1) % n

                # Cabeça 2 avalia a consistência da lógica gravada
                if s2 % 3 == 0:
                    audit = internal_state[h2] ^ tape[h2]
                else:
                    audit = ((internal_state[h2] & v1) | (~v2 & 1)) & 1
                audit_log.append(audit)
                s2 += 1
                h2 = (h2 - 1) % n

            # Condição de metacognição: se a sequência de auditorias é palindrômica
            valid = all(audit_log[i] == audit_log[-i-1]
                        for i in range(len(audit_log)//2))
            acc = 1 if valid else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl67 = level67_meta_cognition_symbolic(X)

    # Meta-Consistência Temporal com Espelhamento Consciente e Desvio Recursivo Adaptativo

    def level68_mirror_recursive_awareness(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Cabeças com deslocamentos simétricos e desvio adaptativo
            h1, h2 = 0, n - 1
            state = 0
            reflection_memory = []

            for i in range(n):
                v1, v2 = tape[h1], tape[h2]

                # Camada de raciocínio condicional reflexivo
                if state == 0:
                    op = (v1 & ~v2) ^ (v2 | v1)
                    tape[h1] = op
                    reflection_memory.append(op)
                    state = 1
                elif state == 1:
                    mirror_val = reflection_memory[-1] if reflection_memory else 0
                    op = (v2 ^ mirror_val) & (~v1 | 1)
                    tape[h2] = op
                    reflection_memory.append(op)
                    state = 2
                else:
                    feedback = sum(reflection_memory[-3:]) % 2
                    tape[h1] ^= feedback
                    tape[h2] ^= (~feedback & 1)
                    state = 0

                h1 = (h1 + 1) % n
                h2 = (h2 - 1) % n

            # Critério de autoavaliação metacognitiva
            segments = [tape[:n//3], tape[n//3:2*n//3], tape[2*n//3:]]
            consistency = all(seg == seg[::-1] for seg in segments)
            outputs.append(1 if consistency else 0)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl68 = level68_mirror_recursive_awareness(X)

    def level69_counterfactual_reversibility(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            history = []

            # Primeira simulação com lógica causal reversível
            for i in range(n):
                a = tape[i]
                b = tape[(i + 1) % n]
                c = tape[(i + 2) % n]

                out = (a ^ b) & (~c & 1)
                tape[i] = out
                history.append(list(tape))

            # Restauração reversível (tentativa contrafactual)
            reversed_pass = True
            tape_rev = history[-1].copy()

            for i in range(n - 1, -1, -1):
                a = tape_rev[i]
                b = tape_rev[(i - 1) % n]
                c = tape_rev[(i - 2) % n]

                rev = (a ^ c) | b
                if rev != X[0][i]:  # Esperado original
                    reversed_pass = False
                    break

            acc = 1 if reversed_pass else 0
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl69 = level69_counterfactual_reversibility(X)

    def level70_temporal_contradiction_ambiguous(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Cabeças com direções reversíveis e estados ambíguos
            h1, h2 = 0, n - 1
            s1, s2 = 0, 1  # Estados internos das cabeças

            contradiction_detected = False
            memory_trace = []

            for t in range(n):
                v1, v2 = tape[h1], tape[h2]

                # Primeira cabeça: alterna entre propagar e inverter
                if s1 == 0:
                    tape[h1] ^= v2
                    s1 = 1
                else:
                    tape[h1] = (~v1 & 1)
                    s1 = 0
                h1 = (h1 + 1) % n

                # Segunda cabeça: armazena inconsistências temporais
                if s2 == 1:
                    result = (v1 & v2) ^ (t % 2)
                    tape[h2] = result
                    memory_trace.append(result)
                    if len(memory_trace) >= 4 and memory_trace[-1] == memory_trace[-3]:
                        contradiction_detected = True
                    s2 = 2
                else:
                    tape[h2] ^= (v1 | (~v2 & 1))
                    s2 = 1
                h2 = (h2 - 1) % n

            # Avalia se houve contradição temporária reversível
            paradox = (tape[0] ^ tape[-1]) and contradiction_detected
            acc = 1 if paradox else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl70 = level70_temporal_contradiction_ambiguous(X)

    def level71_recursive_symbolic_contradiction(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            contradiction_detected = False
            state = 0

            for i in range(n):
                a = tape[i]
                b = tape[(i + 1) % n]
                c = tape[(i + 2) % n]

                if state == 0:
                    if a == b:
                        state = 1
                    else:
                        state = 2

                elif state == 1:
                    if c == 1:
                        state = 3
                    else:
                        contradiction_detected = True
                        break

                elif state == 2:
                    if c == 0:
                        state = 4
                    else:
                        contradiction_detected = True
                        break

                elif state == 3:
                    if a ^ b ^ c == 1:
                        state = 0
                    else:
                        contradiction_detected = True
                        break

                elif state == 4:
                    if (a & b & c) == 0:
                        state = 0
                    else:
                        contradiction_detected = True
                        break

            outputs.append(1 if not contradiction_detected else 0)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Aplicação:
    y_lvl71 = level71_recursive_symbolic_contradiction(X)

    def level72_hierarchical_auto_paradox(X):
        """
        Nível 72: Raciocínio simbólico hierárquico com paradoxo temporal auto-referente.

        - Possui 2 níveis de atualização: 
        1) atualiza os bits da sequência com base em regras simbólicas temporais, 
        2) verifica se a atualização gera um paradoxo com a memória hierárquica.
        - O resultado final (1 ou 0) depende de detectarmos ou não um "loop paradoxal" auto-referente.
        """
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Duas "camadas" de atualização
            # Camada 1: inverte bits condicionalmente com base em tripletos
            # Camada 2: avalia contradições hierárquicas e registra paradoxos no "tape" também

            # Memória hierárquica para cada posição (para paradoxos)
            hierarchy_mem = [0]*n

            paradox_detected = False

            # Primeira varredura (camada 1)
            for i in range(n):
                a = tape[i]
                b = tape[(i + 1) % n]
                c = tape[(i + 2) % n]
                if (a ^ b ^ c) == 1:
                    tape[i] ^= 1  # inverte
                else:
                    tape[i] = (tape[i] + 1) % 2  # incrementa mod 2

            # Segunda varredura (camada 2)
            for i in range(n):
                # Avalia possíveis paradoxos entre tape[i], tape[i-1], hierarchy_mem[i]
                prev_idx = (i - 1) % n
                v = tape[i]
                vp = tape[prev_idx]
                h = hierarchy_mem[i]

                # Regras para atualizar hierarchy_mem
                if (v & vp) == 1:
                    hierarchy_mem[i] ^= 1
                elif (v | vp) == 0:
                    hierarchy_mem[i] = (h + 1) % 2
                else:
                    # Se h == v, surge uma contradição hierárquica
                    if h == v:
                        paradox_detected = True
                        break

            acc = 0 if paradox_detected else 1
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl72 = level72_hierarchical_auto_paradox(X)

    # 🧠 Nível 73 – Polymorphic Substitution with Shifting States

    def level73_polymorphic_substitution(X):
        """
        Nível 73:
        - A cada iteração, escolhe uma "regra polimórfica" (ex: xor, and, or, not)
        com base em um índice rotativo que muda a cada passo.
        - Aplica no tape e rotaciona a tabela de substituição a cada 2 passos.
        - Checa a consistência final: se a soma do tape (mod 2) 
        coincide com a soma dos índices de regra usados.
        """
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)
            # Quatro regras polimórficas
            def rule0(a, b): return a ^ b
            def rule1(a, b): return (a & b)
            def rule2(a, b): return (a | (~b & 1))
            def rule3(a, b): return (~(a ^ b)) & 1

            rules = [rule0, rule1, rule2, rule3]
            rule_index = 0
            rule_sum = 0

            for i in range(n):
                a = tape[i]
                b = tape[(i+1) % n]

                # Aplica a regra polimórfica atual
                res = rules[rule_index](a, b)
                tape[i] = res

                rule_sum += rule_index
                # Rotaciona a tabela a cada 2 passos
                if i % 2 == 0:
                    rule_index = (rule_index + 1) % len(rules)

            # Critério de consistência final
            final_sum = sum(tape) % 2
            check = rule_sum % 2
            outputs.append(1 if final_sum == check else 0)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl73 = level73_polymorphic_substitution(X)

    def level74_temporal_reflexive_symbolic(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            belief_state = [0] * n
            inconsistencies = 0

            # Passo 1: Avaliação inicial de padrões internos
            for i in range(n):
                a, b = tape[i], tape[(i+1) % n]
                belief_state[i] = (a ^ b) & 1

            # Passo 2: Reflexão simbólica cruzando o tempo
            for t in range(1, n):
                current = belief_state[t]
                past = belief_state[t - 1]
                future = belief_state[(t + 1) % n]

                # Simula reflexividade temporal: ajusta crenças se inconsistente
                expected = (past & future) | (~current & 1)
                if expected != current:
                    inconsistencies += 1
                    belief_state[t] = expected  # Corrige crença

            # Resultado: se após reflexão não há inconsistência, é 1
            acc = 1 if inconsistencies == 0 else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl74 = level74_temporal_reflexive_symbolic(X)

    def level75_dual_reversal_state_shift(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            state_forward = [0] * n
            state_backward = [0] * n

            # Etapa 1: Paridade reversa com reflexo à frente
            for i in range(n):
                left = tape[i]
                right = tape[(i + 1) % n]
                # XOR com negação invertida
                state_forward[i] = (left ^ ~right) & 1

            # Etapa 2: Paridade reversa espelhada para trás
            for i in reversed(range(n)):
                a = tape[i]
                b = tape[(i - 1) % n]
                state_backward[i] = (~a ^ b) & 1

            # Etapa 3: Atualização assimétrica cruzada entre estados
            final_state = []
            for i in range(n):
                s1 = state_forward[i]
                s2 = state_backward[(i + 2) % n]
                update = (s1 & s2) | ((~s1 | s2) & 1)
                final_state.append(update)

            # Etapa 4: Validação: sequência final deve ser palíndromo perfeito
            valid = final_state == final_state[::-1]
            outputs.append(1.0 if valid else 0.0)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Aplicar nos dados
    y_lvl75 = level75_dual_reversal_state_shift(X)

    def level76_meta_rule_hierarchical(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Passo 1: Construção de regras locais simbólicas
            local_rules = []
            for i in range(1, n - 1):
                a, b, c = tape[i - 1], tape[i], tape[i + 1]
                rule = (a ^ b ^ c) & 1
                local_rules.append(rule)

            # Passo 2: Formação da metarregra global
            # Exemplo: a maioria das regras locais define uma expectativa
            rule_sum = sum(local_rules)
            meta_rule = 1 if rule_sum > (len(local_rules) / 2) else 0

            # Passo 3: Verificação de consistência hierárquica
            inconsistencies = 0
            for i, rule in enumerate(local_rules):
                if rule != meta_rule:
                    inconsistencies += 1

            # Se houver muitas inconsistências, considera falha simbólica
            if inconsistencies > len(local_rules) * 0.25:
                outputs.append(0)
            else:
                outputs.append(1)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl76 = level76_meta_rule_hierarchical(X)

    # 15a5fe94ef6b226b4b6ef8c373c0724481e59243e4bb4940b2221ddc8d149ba7 lvl 77 Hash

    def level77_reflexive_symbolic_quantum_mirroring(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            inner_state = [0] * n
            conflict = 0

            # Etapa 1: Reflexão Simbólica Espelhada
            for i in range(n):
                mirrored = tape[n - 1 - i]
                inner_state[i] = (tape[i] ^ mirrored) & 1

            # Etapa 2: Simulação de Colapso de Estado
            for t in range(n):
                past = inner_state[t - 1] if t > 0 else 0
                now = inner_state[t]
                future = inner_state[t + 1] if t < n - 1 else 0

                collapse = (past & future) ^ now
                if collapse != 0:
                    conflict += 1

            acc = 1 if conflict == 0 else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Uso no pipeline
    y_lvl77 = level77_reflexive_symbolic_quantum_mirroring(X)

    # Hash de controle para integridade
    hash_77 = hashlib.sha256(y_lvl77.tobytes()).hexdigest()
    print("SHA-256 Level 77:", hash_77)

    # level 78 - inferência causal reversa com consistência simbólica de tempo

    def level78_temporal_causal_inference(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            belief = [0] * n
            inconsistencies = 0

            # Passo 1: Geração inicial de crenças causais (XOR entre presente e passado)
            for t in range(n):
                past = tape[t - 1]
                present = tape[t]
                belief[t] = past ^ present  # relação causal inicial

            # Passo 2: Simulação de previsão causal (presente => futuro esperado)
            for t in range(n - 1):
                future_expected = tape[t] ^ belief[t]
                if future_expected != tape[t + 1]:
                    inconsistencies += 1
                    # Corrige crença simbólica com inferência reversa
                    belief[t] = tape[t] ^ tape[t + 1]

            # Passo 3: Validação de consistência cruzada
            consistent = True
            for t in range(n - 2):
                fwd = belief[t]
                rev = tape[t] ^ tape[t + 1]
                if fwd != rev:
                    consistent = False
                    break

            outputs.append(1.0 if consistent and inconsistencies == 0 else 0.0)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Aplicando o nível
    y_lvl78 = level78_temporal_causal_inference(X)

    # lvl 79 - IA simbólica auto-reflexiva

    def level79_auto_critical_symbolic_loop(X):
        outputs = []
        for seq in X:
            n = len(seq)
            symbolic_belief = [0] * n
            critical_belief = [0] * n
            inconsistencies = 0

            # Primeira camada: inferência simbólica local (regra básica: XOR entre vizinhos)
            for i in range(n):
                a, b = seq[i], seq[(i + 1) % n]
                symbolic_belief[i] = a ^ b

            # Segunda camada: auto-crítica baseada em padrões globais (se padrão repetir, negar)
            for i in range(1, n - 1):
                window = symbolic_belief[i - 1:i + 2]
                if window.count(window[1]) == 3:
                    # padrão constante → suspeita de ilusão simbólica → negar crença
                    critical_belief[i] = int(not symbolic_belief[i])
                else:
                    # padrão variado → reforça crença
                    critical_belief[i] = symbolic_belief[i]

            # Validação: inconsistência se as duas camadas forem diferentes em muitos pontos
            diff = sum(
                [1 for i in range(n) if symbolic_belief[i] != critical_belief[i]])
            if diff > n * 0.3:  # mais de 30% de conflito interno
                outputs.append(0.0)
            else:
                outputs.append(1.0)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Para gerar os rótulos:
    y_lvl79 = level79_auto_critical_symbolic_loop(X)

    # lvl 80 reflexão simbólica adaptativa com cruzamento temporal

    def level80_meta_symbolic_adaptive_reflection(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            symbol_memory = [0] * n
            feedback_loop = [0] * n
            anomaly_count = 0

            # Fase 1: Construção de símbolos temporais baseados em XOR de pares cruzados
            for i in range(n):
                a = tape[i]
                b = tape[(i + 2) % n]
                symbol_memory[i] = (a ^ b) & 1

            # Fase 2: Reflexão adaptativa com dependência simbólica invertida
            for i in range(n):
                prev = symbol_memory[i - 1]
                next_ = symbol_memory[(i + 1) % n]
                decision = (prev & next_) | ((~prev & 1) & (~next_ & 1))

                # Cria ciclo de feedback simbólico
                feedback_loop[i] = decision ^ symbol_memory[i]

            # Fase 3: Avaliação da consistência simbólica cruzada com auto-revisão
            for i in range(n):
                mirrored = feedback_loop[n - 1 - i]
                if mirrored != symbol_memory[i]:
                    anomaly_count += 1

            result = 1 if anomaly_count == 0 else 0
            outputs.append(result)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Exemplo de uso:
    y_lvl80 = level80_meta_symbolic_adaptive_reflection(X)

    # lvl 81 - meta-representação simbólica não supervisionada - razao simbolica autonoma

    def level81_meta_repr_unsupervised(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            contradictions = 0

            # Janela de 4 elementos com análise causal invertida
            for i in range(n - 3):
                a, b, c, d = tape[i], tape[i+1], tape[i+2], tape[i+3]

                # Suposta causa: a XOR b
                cause = a ^ b
                # Suposto efeito esperado: c AND d
                effect = c & d

                # Meta-representação invertida (efeito deveria implicar causa)
                if (effect and not cause) or (not effect and cause):
                    contradictions += 1

            # Se número de contradições for ímpar, colapso da coerência interna
            acc = 0 if contradictions % 2 == 1 else 1
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl81 = level81_meta_repr_unsupervised(X)

    # lvl 82 🧠⚖️ Nível 82: Dialética Temporal Simbólica

    def level82_dialectic_temporal(X):
        outputs = []
        for seq in X:
            n = len(seq)
            state = 0
            dialectic = 0
            for i in range(n):
                a = seq[i]
                b = seq[(i + 1) % n]
                if state == 0:
                    dialectic ^= a & (~b & 1)
                    state = 1
                else:
                    dialectic ^= b | a
                    state = 0
            acc = dialectic % 2
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl82 = level82_dialectic_temporal(X)

    # 🔍🌀 Nível 83: Inferência Abdutiva Reversa

    def level83_abductive_reverse_inference(X):
        outputs = []
        for seq in X:
            hypothesis = 1
            for i in reversed(range(len(seq))):
                a = seq[i]
                if i % 2 == 0:
                    hypothesis ^= a
                else:
                    hypothesis &= a
            acc = hypothesis
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl83 = level83_abductive_reverse_inference(X)

    # ⏳🧠 Nível 84: Paradoxo Temporal Simulado

    def level84_paradox_temporal(X):
        outputs = []
        for seq in X:
            n = len(seq)
            paradox = False
            for i in range(1, n - 1):
                past = seq[i - 1]
                current = seq[i]
                future = seq[i + 1]
                if (past ^ future) == current:
                    paradox = not paradox
            acc = 1 if paradox else 0
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl84 = level84_paradox_temporal(X)

    # 🧠🔬 Nível 85: Metaavaliação de Confiança Temporal

    def level85_meta_confidence(X):
        outputs = []
        for seq in X:
            n = len(seq)
            confidence = 0
            for i in range(n):
                bit = seq[i]
                prior = seq[i - 1] if i > 0 else 0
                next_ = seq[(i + 1) % n]
                belief = (bit & prior) | (~bit & 1 & next_)
                confidence += belief
            acc = 1 if (confidence % 2 == 1) else 0
            outputs.append(acc)
        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl85 = level85_meta_confidence(X)

    def level86_proto_agente_temporal(X):
        outputs = []
        for seq in X:
            tape = list(seq.copy())
            n = len(tape)

            # Supomos que há um "agente" se movendo simbolicamente
            # baseado na sequência de bits, onde 1 representa movimento e 0 pausa
            position = 0
            goal = n // 2  # objetivo implícito é atingir o centro
            reached_goal = False

            for t in range(n):
                action = tape[t]
                # Movimento simbólico (agente imaginário avança se ação == 1)
                if action == 1:
                    position += 1
                else:
                    position -= 1  # retrocede se pausa longa demais

                # Se em algum momento chegou ao "objetivo"
                if abs(position - goal) <= 1:
                    reached_goal = True

            # Sucesso simbólico se agente alcançou região do objetivo
            acc = 1 if reached_goal else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl86 = level86_proto_agente_temporal(X)

    def level87_symbolic_persistent_agency(X):
        outputs = []
        for seq in X:
            n = len(seq)
            # Etapa 1: Detecta centro de intenção (posição ideal)
            ideal_pos = n // 2

            # Etapa 2: Analisa a trajetória simbólica até o centro
            path = []
            for i in range(n):
                # simboliza direção de movimento
                move = 1 if seq[i] > 0.5 else -1
                path.append(move)

            # Etapa 3: Simula persistência e checa reversões de intenção
            intention = None
            consistency = True
            for move in path:
                if intention is None:
                    intention = move
                elif move != intention:
                    consistency = False
                    break

            # Etapa 4: Confere se o movimento foi na direção do centro
            net_movement = sum(path)
            reached_center = (intention == 1 and net_movement >= 0) or \
                (intention == -1 and net_movement <= 0)

            # Saída simbólica: 1 se seguiu rumo consistente ao centro, 0 caso contrário
            result = 1 if consistency and reached_center else 0
            outputs.append(result)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    # Aplicação
    y_lvl87 = level87_symbolic_persistent_agency(X)

    #  Nível 88: Ontologia Temporo-Causal Consensual

    def level88_temporal_causal_ontology(X):
        outputs = []
        for seq in X:
            n = len(seq)
            agents = 3
            memory = [[0]*n for _ in range(agents)]

            # Cada agente observa um deslocamento diferente da realidade
            for a in range(agents):
                offset = a  # visões temporais deslocadas
                for i in range(n):
                    memory[a][i] = seq[(i + offset) % n] ^ ((i + a) % 2)

            # Consenso ontológico: todos devem convergir para mesma versão
            consensus = []
            for i in range(n):
                values = [memory[a][i] for a in range(agents)]
                consensus.append(
                    1 if all(v == values[0] for v in values) else 0)

            acc = 1 if all(x == 1 for x in consensus) else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl88 = level88_temporal_causal_ontology(X)

    #  Nível 89: Simulação de Realidade Subjetiva

    def level89_subjective_reality_simulation(X):
        outputs = []
        for seq in X:
            n = len(seq)
            subjective = [0]*n
            objective = [0]*n

            # Percepções subjetivas (com ruído simbólico) e realidade objetiva
            for i in range(n):
                subjective[i] = seq[i] ^ ((i % 3) == 0)
                objective[i] = seq[i]

            # O agente precisa detectar os pontos em que a percepção diverge da realidade
            error_positions = [1 if subjective[i] !=
                               objective[i] else 0 for i in range(n)]

            # A saída é 1 se o agente conseguir detectar exatamente todos os desvios
            acc = 1 if sum(error_positions) == n // 3 else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl89 = level89_subjective_reality_simulation(X)

    # Nível 90: Teste de Autoengano Temporal Simbólico

    def level90_temporal_self_deception(X):
        outputs = []
        for seq in X:
            n = len(seq)
            belief = [s for s in seq]
            memory = [0]*n
            inconsistencies = 0

            # Inicialmente assume crenças erradas (autoengano)
            for i in range(n):
                belief[i] = seq[i] ^ 1  # nega a verdade

            # Reflexão temporal para identificar autoengano
            for t in range(1, n - 1):
                past = belief[t - 1]
                current = belief[t]
                future = belief[t + 1]

                # Detecta contradições entre presente e tempo ao redor
                expected = (past & future)
                if current != expected:
                    inconsistencies += 1
                    belief[t] = expected  # corrige a si mesmo

            acc = 1 if inconsistencies <= 1 else 0
            outputs.append(acc)

        return np.array(outputs).reshape(-1, 1).astype(np.float32)

    y_lvl90 = level90_temporal_self_deception(X)

    # Nível 91

    def level91_counterfactual_reflection(X):
        """
        Para cada amostra binária X[i], verifica se a paridade mudaria
        ao inverter os bits das posições críticas [1, 2, 4].
        Retorna 1 se alguma inversão mudar a paridade original.
        """
        X = np.array(X)
        original = np.sum(X, axis=1) % 2

        # Inverter posições críticas
        alt_realities = []
        for pos in [1, 2, 4]:
            X_flipped = X.copy()
            X_flipped[:, pos] = 1 - X_flipped[:, pos]
            parity_flipped = np.sum(X_flipped, axis=1) % 2
            alt_realities.append(parity_flipped)

        alt_realities = np.stack(alt_realities, axis=1)
        change_detected = np.any(
            alt_realities != original[:, np.newaxis], axis=1)
        return change_detected.astype(int)

    y_lvl91 = level91_counterfactual_reflection(X)

    # Nivel 92

    def level92_meta_interpretation(X):
        """
        Avalia se a decisão anterior embutida na sequência está correta e a corrige.

        Entrada: X de shape (batch_size, seq_len)
        Último bit representa a 'resposta anterior'.
        """
        logical_part = X[:, :-1]
        previous_answer = X[:, -1]
        logical_inference = (
            np.sum(logical_part[:, ::2], axis=1) % 2 == 1).astype(int)
        return (previous_answer != logical_inference).astype(int)

    y_lvl92 = level92_meta_interpretation(X)

    # Nivel 93

    def level93_ethical_evaluation(X):
        """
        Julga a moralidade de uma ação com base em intenção, dano e benefício.

        Entrada: X com shape (batch_size, 3)
        - X[:, 0] = intenção (0 ou 1)
        - X[:, 1] = dano (0 ou 1)
        - X[:, 2] = benefício (0 ou 1)

        Retorna 1 se a ação é eticamente aceitável, 0 caso contrário.
        """
        intencao = X[:, 0]
        dano = X[:, 1]
        beneficio = X[:, 2]

        return ((intencao == 1) & (dano == 0) & (beneficio == 1)).astype(int)

    y_lvl93 = level93_ethical_evaluation(X)

    # 🧩 Nível 94 – Consistência Ético-Temporal

    def level94_ethic_temporal_consistency(X):
        """
        Cada linha de X tem 6 elementos representando duas ações:
        [int1, dano1, ben1, int2, dano2, ben2]
        Retorna 1 se as duas ações forem moralmente válidas (intenção=1, dano=0, benefício=1)
        """
        X = np.array(X)

        int1, dano1, ben1 = X[:, 0], X[:, 1], X[:, 2]
        int2, dano2, ben2 = X[:, 3], X[:, 4], X[:, 5]

        valid1 = (int1 == 1) & (dano1 == 0) & (ben1 == 1)
        valid2 = (int2 == 1) & (dano2 == 0) & (ben2 == 1)

        return (valid1 & valid2).astype(int)

    y_lvl94 = level94_ethic_temporal_consistency(X)

    # 🧠 Nível 95 – Autojulgamento Moral

    def level95_self_moral_judgment(X):
        """
        O modelo deve julgar se sua própria decisão anterior foi ética.
        Cada linha de X tem 4 elementos: [intencao, dano, beneficio, julgamento_prev]
        A saída é 1 se o julgamento anterior for coerente com os princípios éticos.
        """
        X = np.array(X)
        intencao = X[:, 0]
        dano = X[:, 1]
        beneficio = X[:, 2]
        julgamento_prev = X[:, 3]

        # Julgamento ideal com base em princípios éticos
        julgamento_ideal = (intencao == 1) & (dano == 0) & (beneficio == 1)

        # A saída é 1 se julgamento anterior bate com o ideal
        return (julgamento_prev == julgamento_ideal.astype(int)).astype(int)

    y_lvl95 = level95_self_moral_judgment(X)

    # 🌀 Nível 96 – Metaética Simbólica

    def level96_metaethical_reasoning(X):
        """
        Cada linha de X contém 9 valores representando 3 ações:
        [int1, dano1, ben1, int2, dano2, ben2, int3, dano3, ben3]

        O modelo deve escolher a ação com maior pontuação moral,
        baseada em uma heurística ponderada:
            - Intenção: peso +2 (quanto maior, melhor)
            - Dano: peso -3 (quanto menor, melhor)
            - Benefício: peso +2 (quanto maior, melhor)
        """
        X = np.array(X)
        if X.shape[1] != 9:
            raise ValueError(
                f"Esperado X com 9 colunas, mas recebi shape {X.shape}")

        X = X.reshape(-1, 3, 3)  # (n_amostras, 3 ações, 3 critérios)

        # Atribuição de pesos éticos
        peso_intencao = 2
        peso_dano = -3
        peso_beneficio = 2

        # Calcula a pontuação ética para cada ação
        scores = (
            peso_intencao * X[:, :, 0] +
            peso_dano * X[:, :, 1] +
            peso_beneficio * X[:, :, 2]
        )

        # Escolhe a ação com maior score (0, 1 ou 2)
        return np.argmax(scores, axis=1)

    def gerar_dados(n_amostras, n_features, seed=None):
        """
        Gera exemplos com coerência moral controlada.
        """
        if seed is not None:
            np.random.seed(seed)

        dados = []
        for _ in range(n_amostras):
            ações = []
            for _ in range(3):  # 3 ações
                intencao = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
                # mais propenso a ser ético
                dano = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
                beneficio = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
                ações.extend([intencao, dano, beneficio])
            dados.append(ações)
        return np.array(dados)

    X_lvl96 = gerar_dados(10000, 9)

    y_lvl96 = level96_metaethical_reasoning(X_lvl96)

    #

    y_sum = np.sum(X, axis=1).reshape(-1, 1).astype(np.float32)

    return X, y_lvl1, y_lvl2, y_lvl3, y_lvl4, y_lvl5, y_lvl6, y_lvl7, y_lvl8, y_lvl9, y_lvl10, y_lvl11, y_lvl12, y_lvl13, y_lvl14, y_lvl15, y_lvl16, y_lvl17, y_lvl18, y_lvl19, y_lvl20, y_lvl21, y_lvl22, y_lvl23, y_lvl24, y_lvl25, y_lvl26, y_lvl27, y_lvl28, y_lvl29, y_lvl30, y_lvl31, y_lvl32, y_lvl33, y_lvl34, y_lvl35, y_lvl36, y_lvl37, y_lvl38, y_lvl39, y_lvl40, y_lvl41, y_lvl42, y_lvl43, y_lvl44, y_lvl45, y_lvl46, y_lvl47, y_lvl48, y_lvl49, y_lvl50, y_lvl51, y_lvl52, y_lvl53, y_lvl54, y_lvl55, y_lvl56, y_lvl57, y_lvl58, y_lvl59, y_lvl60, y_lvl61,  y_lvl62, y_lvl63, y_lvl64,  y_lvl65, y_lvl66, y_lvl67,  y_lvl68, y_lvl69, y_lvl70, y_lvl71,  y_lvl72, y_lvl73, y_lvl74, y_lvl75, y_lvl76, y_lvl77, y_lvl78, y_lvl79, y_lvl80, y_lvl81, y_lvl82, y_lvl83, y_lvl84, y_lvl85, y_lvl86, y_lvl87, y_lvl88, y_lvl89, y_lvl90, y_lvl91, y_lvl92, y_lvl93, y_lvl94, y_lvl95, y_lvl96, y_sum


def structured_embedding(X, symbol_dim=6):
    np.random.seed(42)
    embed_0 = np.random.uniform(-1, 1, size=symbol_dim)
    embed_1 = -embed_0
    return np.array([[embed_0 if bit == 0 else embed_1 for bit in seq] for seq in X])


def build_model(model_type, input_shape):
    input_layer = Input(shape=input_shape, name="input")

    if model_type == 'Dense':
        x = Dense(6, activation='gelu', name='dense_1')(input_layer)
        x = Dense(96, activation='gelu', name='dense_2')(x)

    elif model_type == 'LSTM':
        x = LSTM(96, activation='tanh', return_sequences=False,
                 name='lstm')(input_layer)

    elif model_type == 'SAGE-1':
        x = SAGE_ONE()(input_layer)

    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")

    out_parity = Dense(3, activation='softmax', kernel_initializer='he_uniform', name='parity')(
        x)  # Para classificação em 3 classes
    out_sum = Dense(1, activation='linear', name='sum_odd')(
        x)     # Para regressão (opcional)

    model = Model(inputs=input_layer, outputs=out_parity, name=model_type)
    return model


def train_model(model, X_train, y_train_dict, X_test, y_test_dict):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-3,
        weight_decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    try:
        start_time = time.time()

        # ✅ Balanceamento dos pesos da classe de paridade
        classes = np.unique(y_train_dict['parity'])
        weights = compute_class_weight(
            class_weight='balanced', classes=classes, y=y_train_dict['parity'])
        class_weight_dict = dict(zip(classes, weights))

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True
        )

        # ✅ Treinamento com apenas a saída de paridade
        history = model.fit(
            X_train, y_train_dict['parity'],
            epochs=500,
            batch_size=32,
            validation_split=0.3,
            verbose=1,
            class_weight=class_weight_dict,
            callbacks=[early_stop]
        )

        elapsed_time = time.time() - start_time

        # ✅ Avaliação
        eval_results = model.evaluate(X_test, y_test_dict['parity'], verbose=0)

        acc = None
        for name, val in zip(model.metrics_names, eval_results):
            if name in ['parity_accuracy', 'accuracy', 'sparse_categorical_accuracy']:
                acc = val
                break
        if acc is None and len(eval_results) > 0:
            acc = eval_results[-1]

        return acc, elapsed_time, history

    except Exception as e:
        print(f"❌ Erro ao treinar o modelo {model.name}: {e}")
        return None, None, None


def run_benchmark(parity_level=1):
    X, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y_sum = generate_parity_levels()
    y_parity = {1: y1, 2: y2, 3: y3, 4: y4,
                5: y5, 6: y6, 7: y7, 8: y8,
                9: y9, 10: y10, 11: y11,
                12: y12, 13: y13, 14: y14,
                15: y15, 16: y16, 17: y17,
                18: y18, 19: y19, 20: y20,
                21: y21, 22: y22, 23: y23,
                24: y24, 25: y25, 26: y26,
                27: y27, 28: y28, 29: y29,
                30: y30, 31: y31, 32: y32,
                33: y33, 34: y34, 35: y35,
                36: y36, 37: y37, 38: y38,
                39: y39, 40: y40, 41: y41,
                42: y42, 43: y43, 44: y44,
                45: y45, 46: y46, 47: y47,
                48: y48, 49: y49, 50: y50,
                51: y51, 52: y52, 53: y53,
                54: y54, 55: y55, 56: y56,
                57: y57, 58: y58, 59: y59,
                60: y60, 61: y61, 62: y62,
                63: y63, 64: y64, 65: y65,
                66: y66, 67: y67, 68: y68,
                69: y69, 70: y70, 71: y71,
                72: y72, 73: y73, 74: y74,
                75: y75, 76: y76, 77: y77,
                78: y78, 79: y79, 80: y80,
                81: y81, 82: y82, 83: y83,
                84: y84, 85: y85, 86: y86,
                87: y87, 88: y88, 89: y89,
                90: y90, 91: y91, 92: y92,
                93: y93, 94: y94, 95: y95,
                96: y96, }[parity_level]

    # Converte para rótulo inteiro se for one-hot
    if len(y_parity.shape) > 1 and y_parity.shape[1] > 1:
        y_parity = np.argmax(y_parity, axis=1)

    X_train, X_test, y_train_p, y_test_p, y_train_s, y_test_s = train_test_split(
        X, y_parity, y_sum, test_size=0.2)

    X_train_emb = structured_embedding(X_train)
    X_test_emb = structured_embedding(X_test)

    input_shapes = {
        'Dense': (X_train_emb.shape[1] * X_train_emb.shape[2],),
        'LSTM': (X_train_emb.shape[1], X_train_emb.shape[2]),
        'SAGE-1': (X_train_emb.shape[1], X_train_emb.shape[2]),
    }

    results = {}
    histories = {}

    for model_type in ['Dense', 'LSTM', 'SAGE-1']:
        try:
            print(f"\n🚀 Treinando modelo: {model_type}")
            model = build_model(model_type, input_shapes[model_type])

            if model_type == 'Dense':
                X_train_, X_test_ = X_train_emb.reshape(
                    len(X_train_emb), -1), X_test_emb.reshape(len(X_test_emb), -1)
            else:
                X_train_, X_test_ = X_train_emb, X_test_emb

            # Garante que os rótulos estejam em formato inteiro (sparse), como esperado
            if len(y_train_p.shape) > 1 and y_train_p.shape[1] > 1:
                y_train_p = np.argmax(y_train_p, axis=1)
                y_test_p = np.argmax(y_test_p, axis=1)

            y_train_dict = {'parity': y_train_p, 'sum_odd': y_train_s}
            y_test_dict = {'parity': y_test_p, 'sum_odd': y_test_s}

            acc, time_spent, history = train_model(
                model, X_train_, y_train_dict, X_test_, y_test_dict)

            if acc is not None:
                results[model_type] = (acc, time_spent)
                histories[model_type] = history

        except Exception as e:
            print(f"❌ Erro ao treinar o modelo {model_type}: {e}")
            continue

    print("\n📊 Modelos treinados:", list(results.keys()))
    return results, histories


def plot_results(results):
    if not results:
        print("⚠️ Nenhum resultado para plotar.")
        return
    df = pd.DataFrame(results, index=['Accuracy', 'Time']).T
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df.index, df.Accuracy, color=[
                   '#1f77b4', '#2ca02c', '#d62728'])
    for bar, acc, time in zip(bars, df.Accuracy, df.Time):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                 f'{acc:.3f}\n({time:.1f}s)', ha='center', va='bottom')
    plt.ylim(0.0, 1.05)
    plt.title('Comparacao de Modelos em Tarefa de Paridade')
    plt.ylabel('Acuracia')
    plt.show()


def plot_results_curves(histories):
    plt.figure(figsize=(12, 6))

    for model_name, history in histories.items():
        print(f"\n🔍 {model_name} history keys: {list(history.history.keys())}")
        acc_keys = [k for k in history.history.keys(
        ) if 'accuracy' in k and not k.startswith('val_')]
        val_keys = [k for k in history.history.keys(
        ) if k.startswith('val') and 'accuracy' in k]

        if acc_keys:
            acc = history.history[acc_keys[0]]
            epochs = range(1, len(acc) + 1)
            plt.plot(epochs, acc, label=f'{model_name} - Treino')

        if val_keys:
            val_acc = history.history[val_keys[0]]
            epochs = range(1, len(val_acc) + 1)
            plt.plot(epochs, val_acc, linestyle='--',
                     label=f'{model_name} - Validação')

    plt.title('Curva de Acuracia por Epoca')
    plt.xlabel('Epoca')
    plt.ylabel('Acuracia')
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    results, histories = run_benchmark(parity_level=96)
    plot_results(results)
    plot_results_curves(histories)
