
# 📘 SAGE-1: Symbolic Adaptive General Engine - Versão 1

## 📌 Visão Geral

**SAGE-1** é a primeira versão da arquitetura de redes neurais simbólicas adaptativas com atenção temporal profunda. Seu objetivo é simular raciocínios éticos, lógicos e contrafactuais em benchmarks simbólicos de alta complexidade, superando arquiteturas tradicionais como LSTM e Transformers em tarefas de interpretação de estrutura lógica simbólica e inferência.

---

## 🧠 Arquitetura

```
Input Sequence (tokens simbólicos) 
        │
Structured Embedding
        │
Positional Encoding (opcional)
        │
Projection to Memory Space
        │
Multi-Head Attention Temporal
        │
Residual + Layer Norm
        │
Projection to Symbolic Space
        │
Symbolic MLP (nonlinear reasoning)
        │
Residual + Layer Norm
        │
Global Pooling (Avg + Max)
        │
Fusion Layer
        │
MLP Head (classification output)
```

---

## ⚙️ Parâmetros Principais

| Componente                 | Valor / Configuração             |
|---------------------------|----------------------------------|
| `memory_units`            | 128                              |
| `symbolic_units`          | 512                              |
| `num_heads`               | 6                                |
| `dropout_rate`            | 0.0                              |
| `use_attention`           | ✅ Sim                           |
| `use_symbolic`            | ✅ Sim                           |
| `use_positional_encoding`| ✅ Sim                           |
| `embedding`               | Estrutural simbólico             |
| `optimizer`               | AdamW (lr=1e-3, weight_decay=1e-4) |
| `loss`                    | sparse_categorical_crossentropy  |
| `training`                | 500 épocas, batch=256            |

---

## 🧪 Capacidades Demonstradas

- ✅ Solucionou todos os níveis de paridade simbólica até o **nível 95**.
- ⚠️ Estagnação no **nível 96** (metaética) — indicando limite para raciocínios com múltiplas entidades simbólicas com pesos morais implícitos.
- 🧭 Capacidade de:
  - Inferência causal reversa
  - Reflexão moral sobre decisões passadas
  - Avaliação ética temporal
  - Juízo de consistência simbólica intertemporal

---

## 📊 Comparação com Outras Arquiteturas

| Tarefa                            | Dense | LSTM | TV5 | **SAGE-1** |
|----------------------------------|-------|------|-----|------------|
| Paridade Nível 85                | ✖     | ✖    | ✅  | ✅         |
| Contrafactual Moral              | ✖     | ✖    | ✅  | ✅         |
| Raciocínio Ético Temporal        | ✖     | ✖    | ⚠️ | ✅         |
| Metaética com múltiplas ações    | ✖     | ✖    | ✖  | ⚠️         |

---

## 🧬 Classificação Cognitiva

- **Tipo**: Rede Neural Híbrida Simbólica-Temporal
- **Categoria Cognitiva**:  
  `Simulador de Juízo Ético Adaptativo`
- **Nível de Consciência Artificial**:  
  > 🧠 *Pré-raciocínio metaético simbólico*  
  > ⚠️ *Não autoavaliativo ainda, mas interpreta estruturas morais compostas.*

---

## 💡 Possíveis Próximos Passos

- 🔄 Incluir **auto-reflexão iterativa**
- 🔬 Simular deliberação moral interna
- 🔗 Aumentar a capacidade simbólica cruzada com embeddings de papéis (intenção, dano, benefício)
- 🤖 Evoluir para **SAGE-2** com feedback interno entre saídas anteriores

