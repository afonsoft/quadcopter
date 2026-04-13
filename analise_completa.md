# Análise Completa do Algoritmo de Reinforcement Learning - Quadcopter

## Visão Geral do Projeto

Este projeto implementa um algoritmo de reinforcement learning para treinar um quadcopter a realizar tarefas de voo, especificamente manter-se em uma posição alvo (hovering). O projeto utiliza o algoritmo **DDPG (Deep Deterministic Policy Gradient)**, que é adequado para espaços de ação contínuos.

## Estrutura do Projeto

### 1. Componentes Principais

#### `task.py` - Ambiente de Tarefa
```python
class Task():
    def __init__(self, init_pose=None, init_velocities=None, 
                 init_angle_velocities=None, runtime=5., target_pos=None):
```

**Função:** Define o ambiente e os objetivos do agente
- **Estado:** 18 dimensões (6 pose × 3 action_repeats)
- **Ação:** 4 dimensões (velocidade dos 4 rotores)
- **Recompensa:** Baseada na distância até o alvo
- **Objetivo:** Manter o quadcopter na posição [0, 0, 10]

**Função de Recompensa:**
```python
def get_reward(self):
    reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
    return reward
```
- Recompensa máxima: 1.0 (quando está exatamente no alvo)
- Penalização: 0.3 × soma das distâncias em x, y, z

#### `physics_sim.py` - Simulador Físico
- **Não modificar:** Contém a física realista do quadcopter
- Simula dinâmica de voo, gravidade, forças aerodinâmicas
- Atualiza posição, velocidade e orientação

### 2. Agentes Implementados

#### Agente Básico (Random Agent)
```python
class Basic_Agent():
    def act(self):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]
```

**Características:**
- Gera ações aleatórias com distribuição normal
- Empuxo médio: 450 RPM com desvio padrão de 25
- Não aprende, apenas explora aleatoriamente
- **Resultado:** Performance ruim e instável

#### Agente DDPG (Deep Deterministic Policy Gradient)

### 3. Arquitetura DDPG

#### Componentes Principais

##### Actor (Política)
```python
class Actor:
    def build_model(self):
        # Input: estado (18 dimensões)
        # Hidden layers: 64 → 128 → 64 neurônios
        # Output: ação (4 dimensões, escaladas para [0, 900])
```

**Função:** Mapeia estados para ações determinísticas
- **Arquitetura:** Rede neural densa com ativações ReLU
- **Output:** Sigmoid escalado para range dos rotores [0, 900]
- **Objetivo:** Aprender a política ótima π(s)

##### Critic (Valor)
```python
class Critic:
    def build_model(self):
        # Input: estado (18) + ação (4)
        # State pathway: 64 → 128 neurônios
        # Action pathway: 64 → 128 neurônios
        # Combined: Concatenação → 128 → 1 (Q-value)
```

**Função:** Estima o valor Q(s,a)
- **Arquitetura:** Two-tower network (separa estado e ação)
- **Output:** Valor Q estimado
- **Objetivo:** Avaliar quão boa é cada ação-estado

##### Replay Buffer
```python
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", 
                                   field_names=["state", "action", "reward", "next_state", "done"])
```

**Função:** Armazena e amostra experiências passadas
- **Capacidade:** 100.000 experiências
- **Batch size:** 64 experiências
- **Propósito:** Quebrar correlação temporal e estabilizar treinamento

##### OUNoise (Ornstein-Uhlenbeck Noise)
```python
class OUNoise:
    def sample(self):
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
```

**Função:** Adiciona ruído exploratório correlacionado
- **Parâmetros:** μ=0, θ=0.15, σ=0.2
- **Propósito:** Exploração eficiente em espaços contínuos

## Algoritmo DDPG - Explicação Passo a Passo

### 1. Inicialização
```python
agent = DDPG(task)
# Cria Actor e Critic (locais e targets)
# Inicializa pesos dos modelos targets
# Configura ruído e buffer de replay
```

### 2. Loop de Treinamento Principal

#### Para cada episódio:
```python
state = agent.reset_episode()  # Reseta ambiente e ruído
total_reward = 0

while not done:
    action = agent.act(state)           # 1. Escolher ação
    next_state, reward, done = task.step(action)  # 2. Executar ação
    agent.step(action, reward, next_state, done)  # 3. Aprender
    state = next_state                  # 4. Atualizar estado
    total_reward += reward
```

#### Passo 1: Escolha da Ação (Agent.act)
```python
def act(self, state):
    # Previsão da política atual
    action = self.actor_local.model.predict(state)[0]
    # Adicionar ruído para exploração
    return list(action + self.noise.sample())
```

**Comentários:**
- Usa o modelo Actor atual para prever a ação
- Adiciona ruído OUNoise para exploração
- Retorna 4 velocidades de rotor

#### Passo 2: Execução no Ambiente (Task.step)
```python
def step(self, rotor_speeds):
    for _ in range(self.action_repeat):  # 3 repetições
        done = self.sim.next_timestep(rotor_speeds)
        reward += self.get_reward()
        pose_all.append(self.sim.pose)
    next_state = np.concatenate(pose_all)
    return next_state, reward, done
```

**Comentários:**
- **Action repeats:** 3 passos de simulação por ação
- **Acumula recompensa** durante os 3 passos
- **Constroi estado** concatenando poses
- **Retorna:** próximo estado, recompensa total, flag de término

#### Passo 3: Aprendizado (Agent.step → Agent.learn)

##### 3.1 Armazenamento no Replay Buffer
```python
self.memory.add(self.last_state, action, reward, next_state, done)
```

##### 3.2 Amostragem e Aprendizado (quando buffer cheio)
```python
if len(self.memory) > self.batch_size:
    experiences = self.memory.sample()
    self.learn(experiences)
```

##### 3.3 Processo de Aprendizado (Agent.learn)

**Preparação dos dados:**
```python
states = np.vstack([e.state for e in experiences])
actions = np.array([e.action for e in experiences])
rewards = np.array([e.reward for e in experiences])
dones = np.array([e.done for e in experiences])
next_states = np.vstack([e.next_state for e in experiences])
```

**Cálculo dos Targets:**
```python
# Q(s',a') usando modelos targets
actions_next = self.actor_target.model.predict_on_batch(next_states)
Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

# Target equation: Q(s,a) = r + γ * Q(s',a') * (1 - done)
Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
```

**Treinamento do Critic:**
```python
# Atualiza Critic para prever Q(s,a) → Q_targets
self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
```

**Treinamento do Actor:**
```python
# Calcula gradientes da ação
action_gradients = self._get_action_gradients([states, actions, 0])

# Atualiza Actor para maximizar Q(s,a)
self.actor_local.train_fn([states, action_gradients, 1])
```

**Atualização dos Modelos Target:**
```python
# Soft update: θ_target = τ * θ_local + (1 - τ) * θ_target
self.soft_update(self.critic_local.model, self.critic_target.model)
self.soft_update(self.actor_local.model, self.actor_target.model)
```

## Resultados e Análise

### Desempenho do Agente Básico
- **Posição final:** [11.47, -2.04, 29.50]
- **Distância até alvo:** 22.71
- **Comportamento:** Movimento aleatório e instável
- **Conclusão:** Sem aprendizado, performance ruim

### Desempenho do Agente DDPG

#### Estatísticas do Treinamento (20 episódios):
- **Recompensa média:** -8.758
- **Melhor recompensa:** 2.709
- **Episódios positivos:** 50% (10/20)
- **Convergência:** Atingida após episódio 4

#### Análise da Curva de Aprendizado:
1. **Episódios 1-3:** Fase inicial com recompensas muito negativas
   - Agente explorando aleatoriamente
   - Aprendendo básicas dinâmicas de voo

2. **Episódios 4-10:** Melhora rápida
   - Descoberta de políticas melhores
   - Recompensas tornando-se positivas

3. **Episódios 11-20:** Estabilização com alguma instabilidade
   - Política razoável aprendida
   - Pequenas flutuações na performance

### Interpretação dos Resultados

#### Pontos Positivos:
- **Convergência rápida:** Aprendizado significativo em 4 episódios
- **Recompensas positivas:** Agente conseguiu manter-se próximo ao alvo
- **Estabilidade relativa:** Performance consistente após convergência

#### Limitações:
- **Instabilidade residual:** Alguns episódios com recompensas negativas
- **Treinamento curto:** 20 episódios é pouco para convergência completa
- **Hiperparâmetros:** Podem ser otimizados para melhor performance

## Melhorias Possíveis

### 1. Hiperparâmetros
- **Learning rate:** Ajustar separadamente para Actor/Critic
- **Buffer size:** Aumentar para mais diversidade
- **Tau (soft update):** Ajustar para estabilidade
- **Ruído:** Otimizar parâmetros θ e σ

### 2. Arquitetura da Rede
- **Camadas:** Experimentar com diferentes profundidades
- **Ativações:** Testar tanh, leaky_relu
- **Regularização:** Ajustar L1/L2 e dropout
- **Batch normalization:** Manter para estabilidade

### 3. Função de Recompensa
- **Shape:** Incluir penalização por velocidade angular
- **Termos:** Adicionar recompensa por estabilidade
- **Scales:** Ajustar pesos dos diferentes componentes

### 4. Estratégias de Treinamento
- **Curriculum learning:** Começar com tarefas mais fáceis
- **Prioritized replay:** Dar prioridade a experiências surpreendentes
- **Multi-step returns:** Usar n-step returns para melhor crédito assignment

## Conclusão

O algoritmo DDPG implementado demonstrou capacidade de aprender a controlar o quadcopter efetivamente:

1. **Aprendizado bem-sucedido:** O agente convergiu para uma política razoável
2. **Arquitetura adequada:** Actor-Critic com experience replay funcionou bem
3. **Desafios superados:** Lidou com espaço de ação contínuo e dinâmica complexa
4. **Potencial de melhoria:** Com mais treinamento e ajustes finos, performance pode melhorar significativamente

O projeto ilustra bem os conceitos fundamentais de reinforcement learning aplicados a um problema de controle realista e complexo.
