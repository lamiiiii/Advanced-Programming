import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 입력층 개수, 은닉층 깊이, 은닉층 개수, 출력층 개수를 입력 받아
# 가중치의 개수(유전자 길이)를 리턴해주는 함수
def WeightCount(input_count, hidden_depth, hidden_count, output_count):
    count = input_count * hidden_count
    for _ in range(hidden_depth):
        count += hidden_count * hidden_count
    count += hidden_count * output_count
    return count

# 모집단 생성 함수
def MakePopulation(population_size, length):
    chromosomes = []
    for i in range(population_size):
        chromosome = [round(random.uniform(-1, 1), 5) for _ in range(length)]
        chromosomes.append(chromosome)
    return chromosomes

# 전방향 계산 수행하는 함수
def Forward(input, chromosome, hidden_depth, hidden_count, output_count, length):
    inputs = input
    bias = 1
    sum = []

    # 입력층과 첫 번째 은닉층 사이의 가중치 계산
    for i in range(0, hidden_count):
        weighted_sum = bias
        for k in range(len(inputs)):
            weighted_sum += inputs[k] * chromosome[len(inputs) * i + k]
        sum.append(weighted_sum)

    # 추가 은닉층과 출력층 사이의 가중치 계산
    for _ in range(hidden_depth - 1):
        new_sum = []
        hidden_weight = len(inputs) * hidden_count
        for j in range(0, hidden_count):
            weighted_sum = bias
            for k in range(0, len(sum)):
                weighted_sum += sum[k] * chromosome[hidden_weight+ len(sum) * j + k]
            new_sum.append(weighted_sum)
        sum = new_sum

    # 출력층의 가중합 계산 (출력 값이 1개 일 때)
    weighted_sum = bias
    new_sum = []
    ouput_weight = length - hidden_count * output_count

    for j in range(0, len(sum)):
        weighted_sum += sum[j] * chromosome[ouput_weight + j]

    return round(weighted_sum, 5)



# 데이터 불러오기 및 전처리
data = load_breast_cancer()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 입력층 개수를 입력 데이터의 특성 개수로 설정
input_count = X_train.shape[1]


# 에러율 기록
train_error_history = []
best_chromosome_history = []

# 수정된 코드에서 학습과 검증에 사용하는 데이터를 따로 반영
input_count = X_train.shape[1]
hidden_depth = 5  # 은닉층 깊이를 5로 설정
hidden_count = 3  # 은닉층 개수를 3으로 설정
output_count = 1

# NN의 구조를 입력하여 총 가중치의 개수 리턴
length = WeightCount(input_count, hidden_depth, hidden_count, output_count)

# 총 가중치의 개수와 모집단 내 개체수를 매개변수로 주어 모집단 리턴
population_size = 20  # 모집단 크기 증가
chromosomes = MakePopulation(population_size, length)

# GA 학습 시작
iteration_count = 400  # 학습 세대를 50으로 정의


# best_chromosome = None

for G in range(iteration_count):

    train_err = []

    for i, chromosome in enumerate(chromosomes):
        train_predictions = []

        mse_err = 0.0

        for x in X_train:
            train_predictions.append(Forward(x, chromosome, hidden_depth, hidden_count, output_count, length))


        # MSE 손실함수 계산 (하나의 chromosome 별 입력값에 대한 평균 제곱 오차)
        mse_err = round(np.mean((y_train - train_predictions) ** 2), 5)
        train_err.append(mse_err)

    # train_err를 정렬하고 train_errors는 보존
    train_errors = train_err

    # 제일 낮은 에러율을 히스토리 배열에 삽입
    best_err = min(train_errors)
    train_error_history.append(best_err)

    # 각 세대 별 베스트 크로모종 출력 및 베스트 크로모종 히스토리 배열에 삽입
    best_chromosome = chromosomes[train_errors.index(min(train_errors))]
    best_chromosome_history.append(best_chromosome)


    # 토너먼트 셀렉션
    select_count = 4
    if select_count % 2 != 0:
        select_count -= 1

    parents = []
    random_index = random.sample(list(range(population_size)), select_count * 2)
    t = 0.5

    for j in range(0, len(random_index), 2):
        f = random.random()
        p1 = random_index[j]
        p2 = random_index[j+1]
        min_train_err = min(train_errors[p1], train_errors[p2])
        max_train_err = max(train_errors[p1], train_errors[p2])
        if t >= f:
            res = train_errors.index(min_train_err)
            parents.append(chromosomes[res])
        else:
            res = train_errors.index(max_train_err)
            parents.append(chromosomes[res])

    # Box crossover
    childs = []
    random.shuffle(parents)

    for i in range(0, len(parents), 2):
        child = []
        for j in range(0, length):
            crossover_res = round(random.uniform(parents[i][j], parents[i+1][j]), 5)
            child.append(crossover_res)
        childs.append(child)

    # Uniform mutation
    mutation_rate = 0.1

    for i in range(len(childs)):
        for j in range(length):
            if random.random() < mutation_rate:
                min_val = -1.0
                max_val = 1.0
                mutation_val = round(random.uniform(min_val, max_val), 5)
                childs[i][j] = mutation_val

    # GENITOR style Replacement
    sort = sorted(train_err, reverse=True)
    for i in range(len(childs)):
        n = train_errors.index(sort[i])
        n_c = childs[i]
        chromosomes[n] = n_c



# 그래프 그리기
x = range(iteration_count)
best_train_error_values =  min(train_error_history)

print(train_error_history)

print("Best Chromosome:")
print(best_chromosome_history[iteration_count-1])
print("Final Training Error Rate:", train_error_history[iteration_count-1])

plt.plot(x, train_error_history, label="Error")
# plt.plot(x, best_train_error_values, label="Training Error")
plt.xlabel('Generation')
plt.ylabel('Error Rate')
plt.title('Error Rate of the Best Chromosome Over Generations')
plt.legend()
plt.show()
