#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void onePointCrossover(const int parent1[], const int parent2[], int* crossoverPoint, int offspring1[], int offspring2[]);

int main() {
    // 두 개의 부모 배열 정의 (길의 10의 배열)
    int parent1[10];
    int parent2[10];
    
    int crossoverPoint;

    // 사용자로부터 parent1[]입력받기
    printf("Enter the first parent array( 10 binary digits, ex)1010101010 ): ");
    for (int i = 0; i < 10; i++) {
        scanf("%1d", &parent1[i]);
    }

    // 사용자로부터 parent2[]입력받기
    printf("Enter the first parent array( 10 binary digits, ex)1010101010 ): ");
    for (int i = 0; i < 10; i++) {
        scanf("%1d", &parent2[i]);
    }

    // one-point crossover 수행
    int offspring1[10];
    int offspring2[10];
    onePointCrossover(parent1, parent2, &crossoverPoint, offspring1, offspring2);

    // 결과 출력
    printf("Parent 1: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", parent1[i]);
    }
    printf("\n");

    printf("Parent 2: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", parent2[i]);
    }
    printf("\n");
    
    printf("Cut point: before index %d", crossoverPoint);
    printf("\n");

    printf("Offspring 1: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", offspring1[i]);
    }
    printf("\n");

    printf("Offspring 2: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", offspring2[i]);
    }
    printf("\n");

    return 0;
}

// OnePointCrossover 함수 구현
void onePointCrossover(const int parent1[], const int parent2[], int* crossoverPoint, int offspring1[], int offspring2[]) {
    // 교차 지점을 random으로 선택
    srand(time(NULL));
    *crossoverPoint = rand() % 10; // 0부터 10 사이의 랜덤한 인덱스

    // 첫 번째 자손 생성
    for (int i = 0; i < *crossoverPoint; i++) {
        offspring1[i] = parent1[i];
    }
    for (int i = *crossoverPoint; i < 10; i++) {
        offspring1[i] = parent2[i];
    }

    // 두 번째 자손 생성 (부모 역순 사용)
    for (int i = 0; i < *crossoverPoint; i++) {
        offspring2[i] = parent2[i];
    }
    for (int i = *crossoverPoint; i < 10; i++) {
        offspring2[i] = parent1[i];
    }
}
