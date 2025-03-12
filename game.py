from deap import creator, base, tools
import random
import pygame
import math
import numpy as np

assets = "./assets/"
carImage = assets+"c.png" # imagem com redução de escala
mapImage = assets+"m.png" # imagem com redução de escala
aiFile = "ai.npy"

# Alterei a resolução devido a problemas com meu monitor
INITPOS = [830*0.7, 920*0.7]
OLDPOS = [820*0.7, 910*0.7]
WIDTH = 1920*0.7
HEIGHT = 1080*0.7
CAR_SIZE_X = 60*0.7
CAR_SIZE_Y = 60*0.7
TEXT_CENTER = (900*0.7, 490*0.7)

BORDER_COLOR = (255, 255, 255, 255)  # Cor que causa colisão ao tocar
# Parametros da rede neural
Ninput = 6  # Entradas
Nhidden = 10  # neuronios camada oculta
Nsaidas = 4  # neurionios saida
# soma total dos pesos e bias
Ntotal = (Ninput*Nhidden+Nhidden)+(Nhidden*Nsaidas+Nsaidas)

# Parametros do algoritmo genético
CrossProb = 0.9  # Probabilidade de crossover
MutProb = 0.2  # Probabilidade de mutação
PontoParada = 300  # Ponto de parada fitness
TamPop = 50  # Tamanho da população
QtdGeracoes = 5  # Quantidade de gerações

##################################################################################
# Configurações do jogo
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Configurações do relógio
# Configurações de fonte e carregamento do mapa
clock = pygame.time.Clock()
alive_font = pygame.font.SysFont("Arial", 20)
# A conversão acelera muito o processamento
game_map = pygame.image.load(mapImage).convert()
map_array = pygame.surfarray.array3d(game_map)

# Criar uma máscara para identificar a pista preta
track_mask = np.all(map_array == [0, 0, 0], axis=-1)


class Car:
    def __init__(self, text="Treinando IA com DEAP"):
        self.text = text
        # Carregar o sprite do carro e rotacioná-lo
        # Convert acelera muito o processamento
        self.sprite = pygame.image.load(carImage).convert()
        self.sprite = pygame.transform.scale(
            self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = INITPOS.copy()  # Posição inicial
        # Começa do lado correto
        self.angle = -180
        self.speed = 0
        self.oldPosition = OLDPOS

        self.speed_set = False  # Flag para velocidade padrão posteriormente

        self.center = [self.position[0] + CAR_SIZE_X / 2,
                       self.position[1] + CAR_SIZE_Y / 2]  # Calcular o centro

        self.radars = []  # Lista para sensores / radares
        self.drawing_radars = []  # Radares a serem desenhados

        self.alive = True  # Booleano para verificar se o carro colidiu

        self.distance = 0  # Distância percorrida
        self.time = 0  # Tempo decorrido
        # Renderizar texto superior

    def draw(self, screen):
        txt_acima = (TEXT_CENTER[0], TEXT_CENTER[1] - 50)  # 50 pixels acima
        texto_superior = alive_font.render(self.text, True, (0, 0, 255))
        texto_superior_rect = texto_superior.get_rect()
        texto_superior_rect.center = txt_acima
        screen.blit(texto_superior, texto_superior_rect)
        screen.blit(self.rotated_sprite, self.position)  # Desenhar sprite
        self.draw_radar(screen)  # OPCIONAL PARA SENSORES

    def draw_radar(self, screen):
        # Opcionalmente desenhar todos os sensores / radares
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # Se qualquer canto tocar a cor da borda -> Colisão
            # Assume um formato retangular
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(
            self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Enquanto não atingir a cor da borda e o comprimento < 300 (máximo) -> continuar avançando
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(
                self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(
                self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calcular distância até a borda e adicionar à lista de radares
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Definir a velocidade para 20 na primeira vez
        # Apenas quando houver 4 nós de saída com velocidade para cima e para baixo
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Obter sprite rotacionado e mover na direção X correta
        # Não permitir que o carro se aproxime menos de 20px da borda
        self.oldPosition[0] = self.position[0]
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 -
                                     self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Aumentar a distância e o tempo
        self.distance += self.speed
        self.time += 1

        # Mesmo para a posição Y
        self.oldPosition[1] = self.position[1]
        self.position[1] += math.sin(math.radians(360 -
                                     self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calcular novo centro
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2,
                       int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calcular os quatro cantos
        # O comprimento é metade do lado
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) *
                    length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) *
                     length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) *
                       length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) *
                        length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]

        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Verificar colisões e limpar radares
        self.check_collision(game_map)
        self.radars.clear()

        # De -90 a 120 com passo de 45 verificar radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Obter distâncias até a borda
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 20)

        return return_values

    def is_alive(self):
        # Função básica de verificação de vida
        return self.alive

    def get_reward(self):
        # Calcular recompensa (talvez mudar?)
        return self.distance / (CAR_SIZE_X / 2)

    def receba(self):
        return self.speed, self.get_data(), self.get_reward()

    def get_reward_ds(self):
        # Recompensa baseada na distância percorrida
        distance_reward = self.distance / (CAR_SIZE_X / 2)

        # Recompensa baseada nos radares (evitar colisões e manter-se centralizado)
        radar_reward = 0
        if self.radars:
            # Média das distâncias dos radares normalizada
            avg_radar = sum([radar[1]
                            for radar in self.radars]) / len(self.radars)
            radar_reward = avg_radar / 50  # Normalização para escala adequada

            # Bônus adicional para o radar frontal (0 graus)
            front_radar = self.radars[2][1]  # Radar frontal está na posição 2
            radar_reward += front_radar / 100  # Maior peso para visão frontal

        # Penalizar velocidade excessivamente baixa
        speed_penalty = -1 if self.speed < 5 else 0

        # Recompensa total combina todos os componentes
        total_reward = distance_reward + radar_reward + speed_penalty

        return total_reward

    def rotate_center(self, image, angle):
        # Rotacionar o retângulo
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def carMovimento(car: Car):
    screen.blit(game_map, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                car.angle += 10
            if event.key == pygame.K_RIGHT:
                car.angle -= 10
            if event.key == pygame.K_UP:
                car.speed += 2
        else:
            car.speed = 0
    if car.is_alive():
        car.update(game_map)

    car.draw(screen)

    # Essa a string que imprime no meio da tela
    text = alive_font.render(str(car.get_data())+" " +
                             str(car.get_reward()), True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = TEXT_CENTER
    screen.blit(text, text_rect)
    # Atualizar a tela
    pygame.display.flip()
    clock.tick(60)
##################################################################################

# rede neural 
def forward(individual, inputs):
    # w1 = 60 pesos, b1 = 10 bias, w2 = 40 pesos, b2 = 4 bias
    W1 = np.array(individual[:60]).reshape(Ninput, Nhidden)
    b1 = np.array(individual[60:70])
    W2 = np.array(individual[70:110]).reshape(Nhidden, Nsaidas)
    b2 = np.array(individual[110:])

    hidden = np.tanh(np.dot(inputs, W1) + b1)  # Função de ativação tanh
    outputs = np.dot(hidden, W2) + b2

    # Função de ativação softmax na camada de saída
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    # retorna as probabilidades de cada ação
    return softmax(outputs)


# Movimentos possíveis
movs = ["nada", pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP]

# Simula o pressionamento de uma tecla no pygame
def simular_tecla(event, key):
    if key == "nada":
        return
    evento = pygame.event.Event(event, {'key': key})
    pygame.event.post(evento)

# Função global para movimentar o carro
def movimenta(car: Car, individual: np.array):  # Recebe o carro e a IA(treinada ou não)
    entrada = car.receba()  # Recebe a velocidade, os dados dos radares e a recompensa
    # pega somente os dados dos radares e a velocidade
    entrada = np.array(entrada[1]+[entrada[0]])
    tecla = forward(individual, entrada)  # Recebe a saída da função softmax
    index = np.argmax(tecla)  # Pega o índice da maior saída
    simular_tecla(pygame.KEYDOWN, movs[index])  # Simula a tecla pressionada
    carMovimento(car)  # Movimenta o carro

# Implementação da função de aptidão
def aptidao(individual):
    individual = np.array(individual)  # transforma a lista em um array
    car = Car()

    actual_reward = -1
    parado = 0  # para forçar o carro a se mover
    while True:
        if car.get_reward() > PontoParada:  # equivalente a 2 voltas
            return (car.get_reward(),)
        if not car.is_alive():
            return (car.get_reward(),)
        movimenta(car, individual)  # movimenta o carro

        if actual_reward == car.get_reward():
            parado += 1
            if parado >= 17:  # se o carro ficar parado por 17 frames retorne a recompensa atual
                return (car.get_reward(),)
        else:
            parado = 0  # caso se mova resete o contador
        actual_reward = car.get_reward()
        # from time import sleep
        # sleep(0.1)

# Função para treinar a IA
def treina():
    print(
        f"\nConfigurações de treino da rede: {Ninput} entradas, {Nhidden} neurônios na camada oculta, {Nsaidas} neurônios de saída")
    print(f"Funções de ativação: camada oculta -> tanh, camada de saída -> softmax")
    print(
        f"Parâmetros evolutivos: população {TamPop}, {QtdGeracoes} gerações, probabilidade de crossover = {CrossProb}, probabilidade de mutação = {MutProb}")
    print(f"Escolha de individuos por maior fitness, cruzamento de dois pontos, mutação gaussiana com desvio padrão 0.1")
    print(
        f"Seleção por torneio com tamanho 5, ponto de parada = {PontoParada}")
    creator.create("FitnessMax", base.Fitness,
                   weights=(1.0,))  # Criação de fitness
    # Criação de indivíduo
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()  # base
    toolbox.register("attr_float", random.uniform, -2, 2)
    # Criação de indivíduo com 114 parâmetros do tipo float(pesos e bias da rede)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, n=Ntotal)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)  # Criação de população

    toolbox.register("evaluate", aptidao)  # Função de avaliação
    toolbox.register("select", tools.selTournament,
                     tournsize=5)  # Função de seleção
    toolbox.register("mate", tools.cxTwoPoint)  # Função de crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0,
                     sigma=0.1, indpb=0.1)  # Função de mutação

    # Define o tamanho de população como 50
    pop = toolbox.population(n=TamPop)
    # Loop que executa por 5 gerações
    for gen in range(QtdGeracoes):  # 5 gerações
        # Identifica indivíduos que ainda não foram avaliados (fitness inválido)
        invalids = [ind for ind in pop if not ind.fitness.valid]
        # Avalia o fitness dos indivíduos inválidos usando a função de avaliação
        fitnesses = map(toolbox.evaluate, invalids)
        # Atribui os valores de fitness calculados aos indivíduos
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit
            if ind.fitness.values[0] > PontoParada:
                ai = np.array(ind)  # Transforma em ndarray
                np.save(aiFile, ai)  # Salva a IA treinada
                print(f"\n\nTreinada com sucesso com {gen} gerações")
                print(f"IA salva como: {aiFile}")
                return

        # Seleciona indivíduos para a próxima geração usando o método definido no toolbox
        offspring = toolbox.select(pop, len(pop))

        # Clona os indivíduos selecionados para evitar modificar os originais
        offspring = list(map(toolbox.clone, offspring))

        # Aplica o operador de crossover em pares de indivíduos
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CrossProb:  # Probabilidade de crossover
                toolbox.mate(child1, child2)  # Realiza o crossover
                # Remove os valores de fitness dos filhos modificados
                del child1.fitness.values
                del child2.fitness.values

        # Inicia o processo de mutação nos offspring
        for mutant in offspring:
            if random.random() <= MutProb:  # Probabilidade de mutação
                # Realiza a mutação
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Substitui a população
        pop[:] = offspring

if __name__ == "__main__":
    # Tente carregar a IA treinada
    try:
        ai = np.load(aiFile)
        car = Car("Percorrendo com IA treinada")
        while True:
            movimenta(car, ai)
    except Exception as e:  # Se não conseguir, treine a IA
        treina()
