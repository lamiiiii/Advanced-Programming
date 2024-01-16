import random  
import numpy as np 

# 입력층 개수, 은닉층 깊이, 은닉층 개수, 출력층 개수를 입력 받아
# 가중치의 개수(유전자 길이)를 리턴해주는 함수 
def WeightCount(input_count, hidden_depth, hidden_count, output_count) :
    count = input_count * hidden_count
    # for _ in range(hidden_depth):
    #     count += hidden_count * hidden_count
    count += hidden_count * output_count
    return count 


# 모집단 생성 함수
def MakePopulation(population_size, length):
    chromosomes = []
    for i in range(population_size):
        chromosome = [round(random.uniform(-1, 1), 5) for _ in range(length)]
        chromosomes.append(chromosome)
    print("================= Chromosomes Before GA =================")
    print()
    for j in range(population_size):        
        print("Chromosome [%d] " %j)
        print(chromosomes[j])
        print()
    return chromosomes 


 # 층의 구조 (입력층 2, 은닉층 깊이 1, 은닉층 개수 3, 출력층 개수 1)


# 전방향 계산 수행하는 함수
def Forward(input, chromosome, hidden_depth, hidden_count, output_count, length):
    inputs = input
    bias = 1
    sum = []

    # 입력층과 첫 번째 은닉층 사이의 가중치 계산
    for i in range(0, hidden_depth):
        weighted_sum = bias
        for k in range(len(inputs)):
            weighted_sum += inputs[k] * chromosome[len(inputs) * i + k]
        sum.append(weighted_sum)

    # 추가 은닉층과 출력층 사이의 가중치 계산
    # for i in range(hidden_depth - 1):
    #     new_sum = []
    #     weight_start = len(inputs) * hidden_count + i * (hidden_count ** 2)
    #     for j in range(0, hidden_count):
    #         weighted_sum = bias
    #         for k in range(0, len(sum)):
    #             weighted_sum += sum[k] * chromosome[weight_start+ len(sum) * j + k]
    #         new_sum.append(weighted_sum)
    #     sum = new_sum

    # 출력층의 가중합 계산 (출력 값이 1개 일 때)
    weighted_sum = bias
    ouput_weight = len(chromosome) - hidden_count * hidden_count - hidden_count * output_count

    for j in range(0, len(sum)):
        weighted_sum += sum[j] * chromosome[ouput_weight + j]

    return round(weighted_sum, 5)



# XOR 문제의 입력과 출력 데이터 정의
inputs = [[0,0], [0,1],[1,0],[1,1]] 
outputs = [0,1,1,0]

# 입력층 개수, 은닉층의 깊이와 노드 개수, 출력층 개수
input_count = len(inputs[0])
hidden_depth = 1
hidden_count = 3
output_count = 1

# 모집단 크기
population_size = 10

results = []
errs = []
errors = []

# NN의 구조를 입력하여 총 가중치의 개수 리턴
length = WeightCount(input_count, hidden_depth, hidden_count, output_count)   

# 총 가중치의 개수와 모집단 내 개체수를 매개변수로 주어 모집단 리턴
chromosomes = MakePopulation(population_size, length)

outputs = np.array(outputs)

# 전방향 계산 출력 
print("================= Chromosomes After Foward =================")
for i, chromosome in enumerate(chromosomes):
    predictions = [] #예측값 배열

    # 각 크로모좀에 대한 입력에 대한 계산 결과
    for input in inputs:
        predictions.append(Forward(input, chromosome, hidden_depth, hidden_count, output_count, length))
    
    # MSE 손실함수 계산 
    mse_err = round(np.mean((outputs - predictions) ** 2), 5)
    errs.append(mse_err)
    print("Error : %.5f " %(mse_err))

# 최소 에러 및 해당하는 크로모좀 출력
print("================= Min Error =================")
errors = errs # 기본 에러 배열 

print("Chromosome [%i]" %errors.index(sorted(errs)[0]))
print(chromosomes[errors.index(sorted(errs)[0])])
print("Min_Error : %.5f" %(sorted(errs)[0]))
print()

# =================================================================
# GA 학습 시작
# Selection
print("================= GA 학습 시작 =================")
print()

select_count = 4
if (select_count % 2 != 0):
    select_count -= 1

random_index=[]
parents=[]      # selection을 통해 선택된 크로모좀


random_index = ((random.sample(list(range(population_size)), select_count*2)))

print()

t = 0.5 #확률

for j in range(0, len(random_index), 2):
    f = random.random()
    p1 = random_index[j]
    p2 = random_index[j+1]

    # 두 부모 중 에러가 작은 것과 큰 것을 찾음
    min_err = min(errors[p1], errors[p2])
    max_err = max(errors[p1], errors[p2])
    if (t >= f):
        res = errors.index(min_err)
        parents.append(chromosomes[res])
    else:
        res = errors.index(max_err)
        parents.append(chromosomes[res])
# =================================================================

# # Box crossover
print("================= Box crossover =================")
print()
childs = []
random.shuffle(parents)

# 두 개체씩 Box Crossover 수행
for i in range(0, len(parents), 2):
    # 자식 크로모종의 유전자를 저장할 배열 
    child = []

    # 같은 인덱스의 가중치 사이의 실수를 정하는 Box Crossover를 수행
    for j in range(0, length):
        crossover_res= round(random.choice([parents[i][j], parents[i+1][j]]), 5)
        child.append(crossover_res)
    childs.append(child)

for i in range(0, len(childs)):
    print("Childs After Box Crossover [%d]" %i)
    print(childs[i])
    print()

    
# =================================================================

print("================= Uniform mutation =================")
print()
# # Uniform mutation
mutation_rate = 0.1

for i in range(len(childs)):
  for j in range(len(childs[i])):
    if (random.random() < mutation_rate):
        # 돌연변이가 발생하면 가중치를 무작위로 변경
        min_val = -1.0
        max_val = 1.0
        mutation_val = round(random.uniform(min_val, max_val), 5)
        childs[i][j] = mutation_val

# 생성된 자식 크로모좀 출력
for i in range(0, len(childs)):
    print("Childs After Uniform Mutation [%d]" %i)
    print(childs[i])
    print()


# =================================================================
print("================= GENITOR style Replacement =================")
print()
# GENITOR style Replacement

# 에러율이 높은 순으로 정렬
sort = sorted(errs, reverse=True)
for i in range(len(childs)):
   # 에러가 높은 순대로 크로모좀을 대체
   n = errors.index(sort[i])
   print("To be Replaced Chromosome [%d] (Error rate : %.5f)" %(n, sort[i])) 
   print(chromosomes[n])
   print()

   # 대체할 자식 크로모좀
   n_c = childs[i]
   chromosomes[n] = n_c
print()

# =================================================================
# GA 이후 population
errs=[]
errors=[]

# 전방향 계산 출력 
for i, chromosome in enumerate(chromosomes):
    results=[]
    
    # 각 크로모좀에 대한 입력에 대한 계산 결과
    for input in inputs:
        results.append(Forward(input, chromosome, hidden_depth, hidden_count, output_count, length))
    # MSE 손실함수 계산 
    mse_err = round(np.mean((outputs - results) ** 2), 5)
    errs.append(mse_err)
errors = errs # 기본 에러 배열 

print("================= Chromosomes After GA  =================")
print()

for i in range(0, population_size):
    print("Chromosomes After GA [%d] (Error : %.5f)" %(i, errs[i]))
    print(chromosomes[i])
    print()

# 현재 크로모좀들의 에러를 기준으로 정렬
sort = sorted(errs)
print("================= Best Chromosomes  =================")
print()
# 가장 에러가 낮은 크로모좀 출력
print("Best Chromosomes (Error: %.5f)" %sort[0])
print(chromosomes[(errors.index(sort[0]))])
print()
print("================= End  =================")


# ========================== 반복
for G in range(10):
    select_count = 4
    if (select_count % 2 != 0):
        select_count -= 1

    random_index=[]
    parents=[]      # selection을 통해 선택된 크로모종

    random_index = ((random.sample(list(range(population_size)), select_count*2)))

    t = 0.5
    for j in range(0, len(random_index), 2):
        f = random.random()
        p1 = random_index[j]
        p2 = random_index[j+1]

        min_err = min(errors[p1], errors[p2])
        max_err = max(errors[p1], errors[p2])

        if (t >= f):
            res = errors.index(min_err)
            parents.append(chromosomes[res])
        else:
            res = errors.index(max_err)
            parents.append(chromosomes[res])

    # Box crossover
    childs = []
    random.shuffle(parents)

    # 두 개체씩 Box Crossover 수행
    for i in range(0, len(parents), 2):
        # 자식 크로모종의 유전자를 저장할 배열 
        child = []

        # 같은 인덱스의 가중치 사이의 실수를 정하는 Box Crossover를 수행
        for j in range(0, length):
            crossover_res= round(random.uniform(parents[i][j], parents[i+1][j]), 5)
            child.append(crossover_res)
        childs.append(child)

    # # Uniform mutation
    mutation_rate = 0.1

    for i in range(len(childs)):
        for j in range(len(childs[i])):
            if (random.random() < mutation_rate):
                min_val = -1.0
                max_val = 1.0
                mutation_val = round(random.uniform(min_val, max_val), 5)
                childs[i][j] = mutation_val

    # GENITOR style Replacement

    sort = sorted(errs, reverse=True) # 에러율이 높은 순으로 정렬
    for i in range(len(childs)):
        n = errors.index(sort[i])
        n_c = childs[i]
        chromosomes[n] = n_c
    
    # 최적의 크로모좀을 저장하는 부분
    best_chromosome_index = errors.index(min(errors))  # 가장 낮은 에러를 갖는 크로모좀의 인덱스
    best_chromosome = chromosomes[best_chromosome_index]  # 최적의 크로모좀
    chromosomes[best_chromosome_index] = best_chromosome  # 최적의 크로모좀을 현재 크로모좀에 반영


    results=[]
    errs=[]
    errors=[]

    # 전방향 계산 출력 
    for i, chromosome in enumerate(chromosomes):
        predictions = [] #예측값 배열
        
        for input in inputs:
            predictions.append(Forward(input, chromosome, hidden_depth, hidden_count, output_count, length))
        # MSE 손실함수 계산 
        mse_err = round(np.mean((outputs - predictions) ** 2), 5)
        errs.append(mse_err)
    errors = errs # 기본 에러 배열 

    sort = sorted(errs)
    print("================= Best Chromosomes[%i]  =================" %(G+1))
    print()
    print("Best Chromosomes[%i] (Error: %.5f)" %(G+1, sort[0]))
    print(chromosomes[(errors.index(sort[0]))])
    print()
    # print("================= End  =================")

