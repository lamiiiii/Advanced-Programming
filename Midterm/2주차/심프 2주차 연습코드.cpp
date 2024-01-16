// one-point crossover 연습

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ARRAY_SIZE 10

void onePointCrossover(int parent1[ARRAY_SIZE], int parent2[ARRAY_SIZE], int* cutPoint, int offspring1[ARRAY_SIZE], int offspring2[ARRAY_SIZE]);
void printArray(int array[10]);

int main() {
	// ARRAY_SIZE 길이의 parent배열 2개 생성
	int parent1[ARRAY_SIZE];
	int parent2[ARRAY_SIZE];

	// cut point 생성
	int cutPoint;

	// ARRAY_SIZE 길이의 offspring 배열 2개 생성
	int offspring1[ARRAY_SIZE];
	int offspring2[ARRAY_SIZE];

	// parent배열 입력받기
	printf("길이 10의 parent1 배열을 입력하세요. (ex. 0101010101)\n");
	printf("Parent1: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		scanf("%1d", &parent1[i]);
	}
	printf("\n");

	printf("길이 10의 parent2 배열을 입력하세요. (ex. 1010101010)\n");
	printf("Parent2: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		scanf("%1d", &parent2[i]);
	}
	printf("\n");

	// onePointCrossover 실행
	onePointCrossover(parent1, parent2, &cutPoint, offspring1, offspring2);

	// 출력
	printf("\nParent1: ");
	printArray(parent1);
	printf("\nParent2: ");
	printArray(parent2);
	printf("\nCut point: before index %d", cutPoint);
	printf("\nOffspring1: ");
	printArray(offspring1);
	printf("\nOffspring2: ");
	printArray(offspring2);
	printf("\n");
}

// onePointCrossover 실행 함수
void onePointCrossover(int parent1[ARRAY_SIZE], int parent2[ARRAY_SIZE], int* cutPoint, int offspring1[ARRAY_SIZE], int offspring2[ARRAY_SIZE]) {
	// onePointCrossover의 cutPoint 무작위 생성
	srand(time(NULL));
	*cutPoint = rand() % 10; // 0부터 9사이의 index 무작위 생성

	// onePointCrossover 실행
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i < *cutPoint) {
			offspring1[i] = parent1[i];
			offspring2[i] = parent2[i];
		}
		else {
			offspring1[i] = parent2[i];
			offspring2[i] = parent1[i];
		}
	}
};

// 배열 출력 함수
void printArray(int array[10]) {
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", array[i]);
	}
};