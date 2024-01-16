// 4주차 Uniform Crossover
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ARRAY_SIZE 10

int main() {
	srand(time(NULL));
	int arr[30][ARRAY_SIZE] = {};
	int count[30] = { 0 };
	int tarr[4] = { 0 };
	int parr1[ARRAY_SIZE];
	int parr2[ARRAY_SIZE];
	int offspring[ARRAY_SIZE];
	int countF[2] = { 0 };
	double random = 0;

	// 배열 길이 10

	// 30개 배열 생성
	for (int k = 0; k < 30; k++) {
		for (int i = 0; i < ARRAY_SIZE; i++) {
			arr[k][i] = rand() % 2;
			if (arr[k][i] == 1) {
				count[k] = count[k] + 1;
			}
		}
	}

	// 토너먼트 할 4개 배열 랜덤 뽑기
	for (int i = 0; i < 4; i++) {
		tarr[i] = rand() % 30;
	}

	// Generated population 출력
	printf("- Generated population\n");
	for (int k = 0; k < 30; k++) {
		printf("%d: ", k);
		for (int i = 0; i < ARRAY_SIZE; i++) {
			printf("%d", arr[k][i]);
		}
		printf(" (f: %d)", count[k]);
		printf("\n");
	}

	// Tournament selection 출력
	printf("\n-Tournament selection \n");
	printf("Parent 1: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", arr[tarr[0]][i]);
	}
	printf(" (f: %d)", count[tarr[0]]);
	printf("\n");

	printf("Parent 2: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", arr[tarr[1]][i]);
	}
	printf(" (f: %d)", count[tarr[1]]);

	if (count[tarr[0]] > count[tarr[1]]) {
		for (int i = 0; i < ARRAY_SIZE; i++) {
			parr1[i] = arr[tarr[0]][i];
		}
		countF[0] = count[tarr[0]];
	}
	else {
		for (int i = 0; i < ARRAY_SIZE; i++) {
			parr1[i] = arr[tarr[1]][i];
		}
		countF[0] = count[tarr[1]];
	}

	printf("\n");
	printf("Parent 3: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", arr[tarr[2]][i]);
	}
	printf(" (f: %d)", count[tarr[2]]);
	printf("\n");

	printf("Parent 4: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", arr[tarr[3]][i]);
	}
	printf(" (f: %d)", count[tarr[3]]);

	if (count[tarr[2]] > count[tarr[3]]) {
		for (int i = 0; i < ARRAY_SIZE; i++) {
			parr2[i] = arr[tarr[2]][i];
		}
		countF[1] = count[tarr[2]];
	}
	else {
		for (int i = 0; i < ARRAY_SIZE; i++) {
			parr2[i] = arr[tarr[3]][i];
		}
		countF[1] = count[tarr[3]];
	}
	printf("\n");


	printf("\n PARENT1: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", parr1[i]);
	}
	printf(" (f: %d)", countF[0]);

	printf("\n PARENT2: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", parr2[i]);
	}
	printf(" (f: %d)", countF[1]);

	int ucount = 0;

	// Uniform crossover
	for (int i = 0; i < 10; i++) {
		double randomValue = (double)rand() / RAND_MAX;
		if (randomValue > 0.5) {
			offspring[i] = parr1[i];
			if (offspring[i] == 1) {
				ucount = ucount + 1;
			}
		}
		else {
			offspring[i] = parr2[i];
			if (offspring[i] == 1) {
				ucount = ucount + 1;
			}
		}
	}

	// Uniform crossover 출력
	printf("\n - Uniform crossover\n");
	printf("Offspring: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", offspring[i]);
	}
	printf("(f: %d)", ucount);

	return 0;
}