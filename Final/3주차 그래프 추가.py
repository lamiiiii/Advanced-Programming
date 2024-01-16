import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    for i in range(hidden_depth - 1):
        new_sum = []
        weight_start = len(inputs) * hidden_count + i * (hidden_count ** 2)
        for j in range(0, hidden_count):
            weighted_sum = bias
            for k in range(0, len(sum)):
                weighted_sum += sum[k] * chromosome[weight_start+ len(sum) * j + k]
            new_sum.append(weighted_sum)
        sum = new_sum

    # 출력층의 가중합 계산 (출력 값이 1개 일 때)
    weighted_sum = bias
    new_sum = []
    ouput_weight = length - hidden_count * output_count

    for j in range(0, len(sum)):
        weighted_sum += sum[j] * chromosome[ouput_weight + j]

    return round(weighted_sum, 5)

#------------------------------------------------------------------
# 정밀도, 재현율, F1점수 계산 함수
def calculate_precision_recall_f1(y_true, y_pred):
    # True Positive, False Positive, False Negative 계산
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    # 정밀도(Precision) 계산
#     정밀도는 양성으로 예측한 샘플 중 실제로 양성인 샘플의 비율.
#     정밀도 = (양성으로 예측한 정확한 수) / (양성으로 예측한 전체 수) = TP / (TP + FP)
#     TP는 진짜 양성 (True Positive)의 수, FP는 거짓 양성 (False Positive)의 수.
    precision = true_positive / (true_positive + false_positive)

    # 재현율(Recall) 계산
#     재현율은 실제 양성 샘플 중에서 양성으로 정확하게 예측한 샘플의 비율.
#     재현율 = (양성으로 예측한 정확한 수) / (실제 양성 전체 수) = TP / (TP + FN)
#     TP는 진짜 양성 (True Positive)의 수, FN은 거짓 음성 (False Negative)의 수.
    recall = true_positive / (true_positive + false_negative)

    # F1 점수(F1 Score) 계산
    # F1 점수는 정밀도(Precision)와 재현율(Recall)의 조화 평균으로 계산된 값으로, 이진 분류 모델의 성능을 측정하는 데 사용되는 지표.:
    # F1 Score=Precision+Recall2⋅Precision⋅Recall​
    # f1_score = 2 * (precision * recall) / (precision + recall)
    if true_positive + false_positive == 0:
        precision = np.nan
    else:
        precision = true_positive / (true_positive + false_positive)

    return precision, recall, f1_score

# 정밀도(Precision)는 양성으로 예측한 샘플 중에서 실제로 양성인 샘플의 비율을 나타내며, FP(False Positive)를 줄이는 데 중점을 둡니다. 정밀도가 높을수록 거짓 양성 비율이 낮아지고, 모델이 양성으로 예측한 샘플 중에서 실제로 양성인 샘플을 높은 정확도로 찾습니다.

# 재현율(Recall)은 실제 양성 샘플 중에서 양성으로 정확하게 예측한 샘플의 비율을 나타내며, FN(False Negative)를 줄이는 데 중점을 둡니다. 재현율이 높을수록 거짓 음성 비율이 낮아지고, 모델이 실제 양성 샘플을 더 많이 찾아냅니다.

# F1 점수는 정밀도와 재현율의 상충 관계를 고려하여 모델의 성능을 종합적으로 평가합니다. 따라서 F1 점수는 정밀도와 재현율 모두 고려하면서 어느 정도 균형을 맞춘 성능 평가 지표로 사용됩니다. 빈도(Frequency) 기반으로 다루는 분류 작업에서 유용하며, 특히 양성 클래스와 음성 클래스의 분포가 불균형할 때 유용합니다.
#------------------------------------------------------------------


# 데이터 불러오기 및 전처리
data = load_breast_cancer()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 에러율 기록
train_error_history = []
best_chromosome_history = []

# 정확도, 정밀도, 재현율, F1 점수를 저장할 배열 추가
accuracy_history = []
precision_history = []
recall_history = []
f1_history = []

# 입력층 개수를 입력 데이터의 특성 개수로 설정
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
iteration_count = 5000  # 학습 세대를 50으로 정의


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

    # # Box crossover
    # childs = []
    # random.shuffle(parents)

    # for i in range(0, len(parents), 2):
    #     child = []
    #     for j in range(0, length):
    #         crossover_res = round(random.uniform(parents[i][j], parents[i+1][j]), 5)
    #         child.append(crossover_res)
    #     childs.append(child)


#------------------------------------------------------------------

    # Extended Box crossover
    childs = []
    random.shuffle(parents)
    alpha = 0.1  # 확장률

    for i in range(0, len(parents), 2):
        child = []
        for j in range(0, length):
            m = min(parents[i][j], parents[i+1][j])
            M = max(parents[i][j], parents[i+1][j])
            em = m - alpha * (M - m)
            eM = M + alpha * (M - m)
            crossover_res = round(random.uniform(em, eM), 5)
            child.append(crossover_res)
        childs.append(child)

#------------------------------------------------------------------

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


    test_predictions = []
    test_err = []

    for x in X_test:
      # prediction = sigmoid(Forward(x, best_chromosome, hidden_depth, hidden_count, output_count, length))
      # test_predictions.append(prediction)


      prediction = Forward(x, best_chromosome, hidden_depth, hidden_count, output_count, length)

      # 시그모이드 함수의 출력값을 기준으로 0 또는 1로 변환
      binary_prediction = 1 if prediction >= 0.5 else 0

      test_predictions.append(binary_prediction)


    # 정확도 계산
    accuracy = np.sum(np.array(test_predictions) == y_test) / len(y_test)
    accuracy_history.append(accuracy)


#------------------------------------------------------------------
# 정밀도, 재현율, F1점수 계산 함수 실행

    y_pred = np.array(test_predictions)
    precision, recall, f1_score = calculate_precision_recall_f1(y_test, y_pred)
    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1_score)

print("accuracy : ", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
#------------------------------------------------------------------


# 그래프 그리기
x = range(iteration_count)
best_train_error_values =  min(train_error_history)

print(train_error_history)

print("Best Chromosome:")
print(best_chromosome_history[iteration_count-1])
print("Final Training Error Rate:", best_train_error_values)



# fig, ax1 = plt.subplots()

# ax1.set_xlabel('Generation')
# ax1.set_ylabel('Error Rate', color='tab:blue')
# ax1.plot(x, train_error_history, label="Training Error", color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# ax2 = ax1.twinx()
# ax2.set_ylabel('Accuracy', color='tab:red')
# ax2.plot(x, accuracy_history, label="Accuracy", color='tab:red')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# fig.tight_layout()
# plt.title('Training Error and Accuracy Over Generations')
# plt.show()

# # plt.plot(x, train_error_history, label="Error")
# # plt.plot(x, accuracy_history, label="Accuracy")
# # # plt.plot(x, best_train_error_values, label="Training Error")
# # plt.xlabel('Generation')
# # plt.ylabel('Error Rate / Accuracy')
# # plt.title('Error Rate of the Best Chromosome Over Generations')
# # plt.legend()
# # plt.show()

# # 정밀도 그래프
# plt.figure()
# plt.plot(x, precision_history, label="Precision", color='green')
# plt.xlabel('Generation')
# plt.ylabel('Precision')
# plt.title('Precision Over Generations')
# plt.show()

# # 재현율 그래프
# plt.figure()
# plt.plot(x, recall_history, label="Recall", color='orange')
# plt.xlabel('Generation')
# plt.ylabel('Recall')
# plt.title('Recall Over Generations')
# plt.show()

# # F1 점수 그래프
# plt.figure()
# plt.plot(x, f1_history, label="F1 Score", color='purple')
# plt.xlabel('Generation')
# plt.ylabel('F1 Score')
# plt.title('F1 Score Over Generations')
# plt.show()

fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.set_xlabel('Generation')
ax1.set_ylabel('Error Rate', color='tab:blue')
ax1.plot(x, train_error_history, label="Training Error", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:red')
ax2.plot(x, accuracy_history, label="Accuracy", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.figure(figsize=(12, 8))

# 정밀도 그래프
plt.plot(x, precision_history, label="Precision", color='green')

# 재현율 그래프
plt.plot(x, recall_history, label="Recall", color='orange')

# F1 점수 그래프
plt.plot(x, f1_history, label="F1 Score", color='purple')

plt.xlabel('Generation')
plt.ylabel('Metrics')
plt.title('Training Metrics Over Generations')
plt.legend()
plt.show()
