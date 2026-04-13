# Quadcopter Reinforcement Learning

Projeto de reinforcement learning para treinar um quadcopter a realizar tarefas de voo utilizando o algoritmo DDPG (Deep Deterministic Policy Gradient).

## 📋 Descrição do Projeto

Este projeto implementa um sistema de aprendizado por reforço para controlar um quadcopter em simulação. O objetivo é treinar o agente a manter-se em uma posição alvo específica (hovering) usando redes neurais profundas.

## 🚁 Visão Geral

### Objetivo Principal
Treinar um agente inteligente para controlar as velocidades dos 4 rotores de um quadcopter e mantê-lo estável na posição alvo `[0, 0, 10]`.

### Algoritmo Utilizado
- **DDPG (Deep Deterministic Policy Gradient)**
- Adequado para espaços de ação contínuos
- Combina Actor-Critic com experience replay

## 📁 Estrutura do Projeto

```
quadcopter/
├── README.md                    # Este arquivo
├── analise_completa.md          # Análise detalhada do algoritmo
├── Quadcopter_Project.ipynb      # Notebook original da Udacity
├── task.py                      # Definição do ambiente/tarefa
├── physics_sim.py               # Simulador físico (não modificar)
├── agents/                      # Agentes de aprendizado
│   ├── agent.py                 # Agente DDPG original
│   ├── simple_ddpg.py           # Versão simplificada e compatível
│   └── policy_search.py         # Agente de exemplo
├── data.txt                    # Dados da simulação com agente básico
├── rewards.txt                 # Dados de recompensas do treinamento
└── training_analysis.png         # Gráficos de análise do treinamento
```

## 🛠️ Configuração do Ambiente

### Pré-requisitos
- Python 3.8+
- TensorFlow 2.x
- Keras 3.x
- NumPy
- Matplotlib

### Instalação
```bash
pip install numpy matplotlib tensorflow keras
```

## 🚀 Como Executar

### 1. Agente Básico (Random)
```python
import numpy as np
from task import Task

# Configuração
runtime = 5.
init_pose = np.array([0., 0., 10., 0., 0., 0.])
init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])

# Execução
task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
agent = Basic_Agent(task)

# Simulação
while not done:
    rotor_speeds = agent.act()
    _, _, done = task.step(rotor_speeds)
```

### 2. Agente DDPG (Treinamento)
```python
import numpy as np
from agents.simple_ddpg import DDPG
from task import Task

# Configuração
num_episodes = 100
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = DDPG(task)

# Treinamento
for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode()
    total_reward = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        total_reward += reward
        agent.step(action, reward, next_state, done)
        state = next_state
        
        if done:
            print(f'Episódio {i_episode}: recompensa = {total_reward:.3f}')
            break
```

## 📊 Resultados Obtidos

### Agente Básico
- **Posição final:** [11.47, -2.04, 29.50]
- **Distância até alvo:** 22.71
- **Comportamento:** Movimento aleatório e instável

### Agente DDPG (20 episódios)
- **Melhor recompensa:** 2.709
- **Recompensa média:** -8.758
- **Taxa de sucesso:** 50% (episódios com recompensa positiva)
- **Convergência:** Atingida após 4 episódios

## 🧠 Arquitetura do Algoritmo

### Actor (Política)
- **Entrada:** Estado (18 dimensões)
- **Camadas ocultas:** 64 → 128 → 64 neurônios
- **Ativação:** ReLU
- **Saída:** Ações (4 velocidades de rotor)

### Critic (Valor)
- **Entrada:** Estado (18) + Ação (4)
- **Arquitetura:** Two-tower (separado para estado e ação)
- **Camadas:** 64 → 128 neurônios (cada pathway)
- **Saída:** Q-value estimado

### Componentes Adicionais
- **Replay Buffer:** 100.000 experiências
- **OUNoise:** Ruído para exploração
- **Target Networks:** Para estabilidade do treinamento

## 📈 Análise de Desempenho

### Curva de Aprendizado
1. **Episódios 1-3:** Fase inicial (-75.6 → -32.3)
2. **Episódios 4-10:** Melhora rápida (-3.6 → 2.7)
3. **Episódios 11-20:** Estabilização

### Métricas Principais
- **Convergência rápida:** Aprendizado significativo em 4 episódios
- **Estabilidade relativa:** Performance consistente após convergência
- **Potencial de melhoria:** Com mais treinamento pode melhorar significativamente

## 🔧 Parâmetros Configuráveis

### Hiperparâmetros do DDPG
```python
# Rede Neural
learning_rate = 0.001
hidden_layers = [64, 128, 64]

# Treinamento
buffer_size = 100000
batch_size = 64
gamma = 0.99  # fator de desconto
tau = 0.01    # soft update

# Ruído
ou_theta = 0.15
ou_sigma = 0.2
ou_mu = 0.0
```

### Parâmetros da Tarefa
```python
# Ambiente
runtime = 5.0
action_repeat = 3
target_pos = [0., 0., 10.]

# Ações
action_low = 0
action_high = 900
action_size = 4
```

## 🎯 Função de Recompensa

```python
def get_reward(self):
    reward = 1. - 0.3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
    return reward
```

- **Recompensa máxima:** 1.0 (exatamente no alvo)
- **Penalização:** 0.3 × soma das distâncias em x, y, z
- **Objetivo:** Minimizar distância até o alvo

## 📝 Melhorias Possíveis

### 1. Hiperparâmetros
- Ajustar learning rates separadamente para Actor/Critic
- Otimizar parâmetros do ruído OUNoise
- Experimentar com diferentes valores de tau

### 2. Arquitetura
- Testar diferentes profundidades de rede
- Experimentar com outras ativações (tanh, leaky_relu)
- Ajustar regularização L1/L2

### 3. Estratégias de Treinamento
- Curriculum learning (tarefas progressivas)
- Prioritized experience replay
- Multi-step returns

### 4. Função de Recompensa
- Incluir penalização por velocidade angular
- Adicionar termos de estabilidade
- Ajustar pesos dos componentes

## 🔬 Análise Detalhada

Para uma análise completa do algoritmo, incluindo:
- Explicação passo a passo de cada função
- Comentários detalhados do código
- Interpretação dos resultados
- Sugestões de melhorias

Consulte o arquivo `analise_completa.md`.

## 📚 Referências

- [DDPG Paper Original](https://arxiv.org/abs/1509.02971)
- [Udacity Robotics Nanodegree](https://www.udacity.com/course/robotics-software-engineer--nd209)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

## 👤 Autor

Projeto baseado no curriculum da Udacity, implementado e analisado como parte do estudo de Inteligência Artificial e Reinforcement Learning.

## 📄 Licença

Este projeto é para fins educacionais e de pesquisa.
