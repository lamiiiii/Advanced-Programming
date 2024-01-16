# 입력층 개수, 은닉층 깊이, 은닉층 개수, 출력층 개수를 입력 받아
# 가중치의 개수(유전자 길이)를 리턴해주는 함수 
def WeightCount(input_count, hidden_depth, hidden_count, output_count) :
    count = input_count * hidden_count
    for _ in range(hidden_depth):
        count += hidden_count * hidden_count
    count += hidden_count * output_count
    return count 




# 모집단 생성 함수
import random   
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
def Forward(input, chromosome, hidden_depth, hidden_count, output_count) : 
    inputs = input 
    bias = 1
    sum = []
    res = 0.0
    
    # # 깊이 + 1개의 가중치 배열 분리 
    # for i in range(hidden_depth):
    #     chromosome[0] .. chromosome[len(input) * hidden_count - 1] # 6개
    #     chromosome[-3] .. chromosome[len(chromosome)] # 3개
    
    # 입력층과 은닉층 사이의 가중치 계산
    for i in range(0, hidden_count * 2 -2, 2): 
        sum.append(input[0] * chromosome[i] + input[1] * chromosome[i+1] + bias)



    # 은닉층과 출력층 사이의 가중치 계산 
    for j in range(len(sum)) :
        res += sum[j] * chromosome[(hidden_count * 2) -1 + j]
    return round(res, 5)




input = [0, 1]
input_count = len(input)
hidden_depth = 1
hidden_count = 3
output_count = 1
population_size = 10
results = []
errs = []
errors = []


# NN의 구조를 입력하여 총 가중치의 개수 리턴
length = WeightCount(input_count, hidden_depth, hidden_count, output_count)   

# 총 가중치의 개수와 모집단 내 개체수를 매개변수로 주어 모집단 리턴
chromosomes = MakePopulation(population_size, length)

# 전방향 계산 출력 
print("================= Chromosomes After Foward =================")
for i, chromosome in enumerate(chromosomes):
    results.append(Forward(input, chromosome, hidden_depth, hidden_count, output_count))
    
    print("Chromosome [%d]" %(i))
    print("forward: %.5f" %(results[i]))

    # MSE 손실함수 계산 
    err = round((1 - results[i]) ** 2, 5)
    errs.append(err)
    print("Error : %.5f " %(err))
    print()
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
parents=[]      # selection을 통해 선택된 크로모종


random_index = ((random.sample(list(range(population_size)), select_count*2)))

print()

t = 0.7
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
        crossover_res= round(random.uniform(parents[i][j], parents[i+1][j]), 5)
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
  for j in range(len(child)):
    if (random.random() < mutation_rate):
        min_val = -1.0
        max_val = 1.0
        mutation_val = round(random.uniform(min_val, max_val), 5)
        childs[i][j] = mutation_val

for i in range(0, len(childs)):
    print("Childs After Uniform Mutation [%d]" %i)
    print(childs[i])
    print()


# =================================================================
print("================= GENITOR style Replacement =================")
print()
# GENITOR style Replacement

sort = sorted(errs, reverse=True) # 에러율이 높은 순으로 정렬
for i in range(len(childs)):
   n = errors.index(sort[i])
   print("To be Replaced Chromosome [%d] (Error rate : %.5f)" %(n, sort[i])) 
   print(chromosomes[n])
   print()
   n_c = childs[i]
   chromosomes[n] = n_c
print()

# =================================================================
# GA 이후 population

results=[]
errs=[]
errors=[]

# 전방향 계산 출력 
for i, chromosome in enumerate(chromosomes):
    results.append(Forward(input, chromosome, hidden_depth, hidden_count, output_count))
    # MSE 손실함수 계산 
    err = round((1 - results[i]) ** 2, 5)
    errs.append(err)
errors = errs # 기본 에러 배열 

print("================= Chromosomes After GA  =================")
print()
for i in range(0, population_size):
    print("Chromosomes After GA [%d] (Error : %.5f)" %(i, errs[i]))
    print(chromosomes[i])
    print()


sort = sorted(errs)
print("================= Best Chromosomes  =================")
print()
print("Best Chromosomes (Error: %.5f)" %sort[0])
print(chromosomes[errors.index(sort[0])])
print()
print("================= End  =================")