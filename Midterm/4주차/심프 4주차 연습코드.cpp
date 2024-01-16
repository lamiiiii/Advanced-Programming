// Tournament selection과 Uniform selection 연습

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ARRAY_SIZE 10

void printArray(int array[ARRAY_SIZE]);

int main() {
	int arrays[30][10]; // 길이 10의 배열 30개
	int fitnessArrays[30] = { 0 }; // 배열 30개의 fitness 값
	// int overlapPrevent[30] = { 0 }; // 중복 방지를 위한 배열
	int parent1[ARRAY_SIZE];
	int parent2[ARRAY_SIZE];
	int offspring[ARRAY_SIZE];
	int rando1, rando2; // 랜덤 배열 2개 뽑기 위한 수
	int parentFit1, parentFit2, offspringFit = 0;

	srand(time(NULL));

	// 1. 길이 10의 랜덤 배열 30개 생성
	for (int i = 0; i < 30; i++) {
		for (int k = 0; k < ARRAY_SIZE; k++) {
			arrays[i][k] = rand() % 2; // 0과 1, 둘 중 하나 랜덤으로 배열
			if (arrays[i][k] == 1) fitnessArrays[i] += 1;
		}
	}

	// 2. 30개의 배열 중 랜덤으로 2개 뽑아서 Tournament selection을 2번 진행
	do {
		rando1 = rand() % 30;
		rando2 = rand() % 30;
	} while (rando1 == rando2);
	if (fitnessArrays[rando1] > fitnessArrays[rando2]) {
		for (int i = 0; i < ARRAY_SIZE; i++) parent1[i] = arrays[rando1][i];
		parentFit1 = fitnessArrays[rando1];
	}
	else {
		for (int i = 0; i < ARRAY_SIZE; i++) parent1[i] = arrays[rando2][i];
		parentFit1 = fitnessArrays[rando2];
	}

		// 두번째 토너먼트
	do {
		rando1 = rand() % 30;
		rando2 = rand() % 30;
	} while (rando1 == rando2);
	if (fitnessArrays[rando1] > fitnessArrays[rando2]) {
		for (int i = 0; i < ARRAY_SIZE; i++) parent2[i] = arrays[rando1][i];
		parentFit2 = fitnessArrays[rando1];
	}
	else {
		for (int i = 0; i < ARRAY_SIZE; i++) parent2[i] = arrays[rando2][i];
		parentFit2 = fitnessArrays[rando2];
	}

	// 3. 랜덤 넘버를 생성해서 0.5보다 크면 첫번째 분모를, 작으면 두번째 분모를 집어넣어 자손 생성
	for (int i = 0; i < ARRAY_SIZE; i++) {
		double randomNumber = (double)rand() / RAND_MAX;
		if (randomNumber > 0.5) offspring[i] = parent1[i];
		else offspring[i] = parent2[i];
		
		if (offspring[i] == 1) offspringFit += 1;
	}

	// 4. 출력
	printf("- Generated population \n"); // 랜덤 배열 30개 생성 및 출력
	for (int i = 0; i < 30; i++) {
		printf("%d: ", i);
		printArray(arrays[i]);
		printf(" (f: %d)\n", fitnessArrays[i]);
	}
	printf("\n- Tournament selection \n"); // tournament selection을 통해 뽑힌 parent배열들 출력
	printf("Parent1: ");
	printArray(parent1);
	printf(" (f: %d)\n", parentFit1);
	printf("Parent2: ");
	printArray(parent2);
	printf(" (f: %d)\n", parentFit2);
	printf("\n- Uniform crossover \n"); // uniform crossover한 offspring 배열 출력
	printf("Offspring: ");
	printArray(offspring);
	printf(" (f: %d)\n", offspringFit);

	return 0;
}

void printArray(int array[ARRAY_SIZE]) { // 배열 출력 함수
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", array[i]);
	}
}