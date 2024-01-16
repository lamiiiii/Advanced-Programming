import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 입력층 개수, 은닉층 깊이, 은닉층 개수, 출력층 개수를 입력 받아
# 가중치의 개수(유전자 길이)를 리턴해주는 함수
def WeightCount(input_count, hidden_depth, hidden_count, output_count):
    # 입력층과 첫 번째 은닉층 사이의 가중치 개수 계산
    count = input_count * hidden_count
    # 추가 은닉층과 출력층 사이의 가중치 개수 계산
    for _ in range(hidden_depth - 1):
        count += hidden_count * hidden_count
    # 출력층의 가중치 개수 계산
    count += hidden_count * output_count
    # 총 가중치 개수(유전자 길이) 반환
    return count

# 모집단 생성 함수
def MakePopulation(population_size, length):
    chromosomes = []  # 모집단을 담을 리스트 생성
    for i in range(population_size):
        # 각 유전자(가중치)를 저장할 리스트를 생성
        chromosome = [round(random.uniform(-1, 1), 5) for _ in range(length)]
        # 유전자의 길이(length)만큼 반복하여, 랜덤한 가중치 값을 생성하고 소수점 다섯 자리까지 반올림하여 리스트에 추가
        chromosomes.append(chromosome)  # 생성된 유전자를 모집단 리스트에 추가
    return chromosomes  # 초기 모집단 반환

# 전방향 계산 수행하는 함수
def Forward(input, chromosome, hidden_depth, hidden_count, output_count, length):
    inputs = input  # 입력 데이터
    bias = 1  # 편향 (bias) 초기화
    sum = []  # 가중합을 저장할 리스트 초기화

    # 입력층과 첫 번째 은닉층 사이의 가중치 계산
    for i in range(0, hidden_count):
        weighted_sum = bias  # 초기화: 편향 값
        for k in range(len(inputs)): #입력층 개수 반복
            weighted_sum += inputs[k] * chromosome[len(inputs) * i + k]  # 입력 데이터와 가중치를 곱하고 편향을 더해 가중합 계산
            # len(inputs) -> 가중치 
        sum.append(weighted_sum)  # 각 은닉 노드의 가중합을 저장 (hidden count만큼)


    # 추가 은닉층과 출력층 사이의 가중치 계산
    for i in range(hidden_depth - 1):
        new_sum = []
        hidden_weight = len(inputs) * hidden_count
        for j in range(0, hidden_count):
            weighted_sum = bias  # 초기화: 편향 값
            for k in range(0, len(sum)):
                weighted_sum += sum[k] * chromosome[hidden_weight + len(sum) * j + k]  # 이전 은닉층의 출력과 가중치를 곱하고 편향을 더해 가중합 계산
            new_sum.append(weighted_sum)  # 각 은닉 노드의 가중합을 저장
        sum = new_sum  # 새로 계산한 가중합을 현재 가중합으로 업데이트

    # 출력층의 가중합 계산
    weighted_sum = bias  # 초기화: 편향 값
    new_sum = []
    ouput_weight = length - hidden_count * output_count # chromosome 시작 지점 찾기 위한 계산

    for j in range(len(sum)):
        weighted_sum += sum[j] * chromosome[ouput_weight + j]  # 최종 출력층의 가중합 계산
    
    return round(weighted_sum, 5)  # 계산 결과를 반환하고, 다섯 자리까지 반올림하여 소수점 처리
    # 출력 값이 1개일 때

# 데이터 불러오기 및 전처리
data = load_breast_cancer() #라이브러리
X, y = data.data, data.target # x는 입력 데이터(특성), y는 입력에 대한 이진 분류 목표 레이블(0 또는 1)
X = StandardScaler().fit_transform(X)  # 데이터 표준화, 평균을 0 표준편차를 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #학습 데이터와 테스트 데이터로 분할

# 에러율 기록
train_error_history = []  # 학습 에러율을 저장할 빈 리스트 초기화
best_chromosome_history = []   # 검증 에러율을 저장할 빈 리스트 초기화

# 입력층 개수를 입력 데이터의 특성 개수로 설정
input_count = X_train.shape[1]   # 입력 데이터의 특성 개수를 변수에 저장
hidden_depth = 5                # 은닉층 깊이를 5로 설정
hidden_count = 3               # 은닉층 개수를 3으로 설정
output_count = 1               # 출력층 개수를 1로 설정


# NN의 구조를 입력하여 총 가중치의 개수 리턴
length = WeightCount(input_count, hidden_depth, hidden_count, output_count)

# 총 가중치의 개수와 모집단 내 개체수를 매개변수로 주어 모집단 리턴
population_size = 20        # 모집단 크기를 20으로 설정
chromosomes = MakePopulation(population_size, length)

# GA 학습 시작
iteration_count = 1000  # 유전 알고리즘의 학습 세대(iteration) 수를 설정

# best_chromosome = None

for G in range(iteration_count):  # 지정된 세대 수만큼 반복

    train_err = []  # 각 개체의 학습 데이터 에러율을 저장할 리스트 초기화

    for i, chromosome in enumerate(chromosomes):  # 모집단 내 각 개체(염색체)에 대해 반복
        train_predictions = []  # 현재 개체로 계산한 학습 데이터 예측 결과를 저장할 리스트 초기화

        mse_err = 0.0

        for x in X_train:  # 학습 데이터 샘플에 대해 반복
            train_predictions.append(Forward(x, chromosome, hidden_depth, hidden_count, output_count, length))  # 전방향 계산으로 학습 데이터 예측 결과 생성


        # MSE 손실함수 계산 (하나의 chromosome 별 입력값에 대한 평균 제곱 오차)
        mse_err = round(np.mean((y_train - train_predictions) ** 2), 5) # 제일 낮은거 mean
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

    # # Box crossover
    # childs = []
    # random.shuffle(parents)

    # for i in range(0, len(parents), 2):
    #     child = []
    #     for j in range(0, length):
    #         crossover_res = round(random.uniform(parents[i][j], parents[i+1][j]), 5)
    #         child.append(crossover_res)
    #     childs.append(child)

#-------------------------------------------------------
# Extended Crossover
childs = []

for i in range(0, len(parents), 2):
    child1 = []
    child2 = []
    
    for j in range(0, length):
        if random.random() < 0.5:
            child1.append(parents[i][j])
            child2.append(parents[i+1][j])
        else:
            child1.append(parents[i+1][j])
            child2.append(parents[i][j])
    
    childs.append(child1)
    childs.append(child2)

#-------------------------------------------------------

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

# 테스트 
test_predictions = []
test_err = []

for x in X_test:
  test_predictions.append(Forward(x, best_chromosome_history[iteration_count-1], hidden_depth, hidden_count, output_count, length))

# MSE 손실함수 계산 (하나의 chromosome 별 입력값에 대한 평균 제곱 오차)
mse_err = round(np.mean((y_test - test_predictions) ** 2), 5)
test_err.append(mse_err)

x = range(len(X_test))

plt.plot(x, test_predictions, label="Error Rate")
plt.xlabel('test datasets')
plt.ylabel('Error Rate')
plt.title('Error Rate of the Best Chromosome Over Generations')
plt.legend()
plt.show()