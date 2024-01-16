// 5주차 과제 및 실습
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ARRAY_SIZE 20
#define GENERATION_SIZE 250 //250번쯤 해야 fitness가 20인 population을 찾아내는 확률이 높아짐

/*
*  - 한 배열의 최대 길이가 20인 문제를 풀기 위한 유전 알고리즘 구현
*  - 각 chromosome의 fitness는 1의 개수
*  - tournament selection
*  - one-point crossover
*  - bit-flip mutation
*  - Replacement
*/
void onePointCrossover(int parent1[ARRAY_SIZE], int parent2[ARRAY_SIZE], int crossoverOffspring[ARRAY_SIZE]); // one-point crossover 실행 함수
void bitflipMutation(int offspring[ARRAY_SIZE]);
void copyArray(int originalArray[ARRAY_SIZE], int copyArray[ARRAY_SIZE]); // array 복사 함수
void printArray(int array[ARRAY_SIZE]); // 배열 출력 함수

int main() {
	srand(time(NULL));
	int arr[30][ARRAY_SIZE] = { 0 }; // 배열 길이가 20인 배열 30개 
	int fitArr[30] = { 0 };
	int bestArr[GENERATION_SIZE][ARRAY_SIZE] = { 0 }; // tournament selection을 통해 뽑힌 배열 GENERATION_SIZE개
	int fitBestArr[GENERATION_SIZE] = { 0 }; // 뽑힌 배열 GENERATION_SIZE개의 fitness

	// 30개 배열에 랜덤으로 0 또는 1 넣기
	for (int k = 0; k < 30; k++) {
		for (int i = 0; i < ARRAY_SIZE; i++) {
			arr[k][i] = rand() % 2;
			if (arr[k][i] == 1) {    // 1의 개수가 각 chromosome의 fitness 
				fitArr[k] = fitArr[k] + 1;
			}
		}
	}

// 1. Tournament Selection
	for (int k = 0; k < GENERATION_SIZE; k++) {	// GENERATION_SIZE번의 tournament를 통해 best 산출

		printf("- Generation %d\n", k);
		for (int a = 0; a < 30; a++) { // 30개 배열 반복 출력
			printf("%d: ", a);
			printArray(arr[a]);
			printf("  (f: %d)\n", fitArr[a]);
		}

		int rando1 = 0, rando2 = 0;
		rando1 = rand() % 30;
		do { rando2 = rand() % 30; } while (rando1 == rando2); // rando1과 rando2 중복방지

// 2. one-point Crossover
		// fitness를 비교하여 tournament selection
		// tournament selection을 통해 더 높은 fitness를 가진 배열이 onepoint crossover함수에 첫번째 배열로 전달
		int offspring[ARRAY_SIZE];
		if (fitArr[rando1] > fitArr[rando2]) { onePointCrossover(arr[rando1], arr[rando2], offspring); }
		else { onePointCrossover(arr[rando2], arr[rando1], offspring); }

// 3. bit-flip Mutation
		// bit-flip mutation 함수 실행 (무작위로 선택해서 0은 1로, 1은 0으로 바꿈)
		bitflipMutation(offspring);
		int offspringFit = 0;
		for (int i = 0; i < ARRAY_SIZE; i++) { if (offspring[i] == 1) offspringFit++; } // offspring의 fitness 저장

// 4. GENITOR style Replacement
		// offspring 배열을 기존 population 중 fitness가 가장 작은 배열과 Replacement함
		int replacementIndex = 0; int replacementFitness = fitArr[0]; // replacement할 배열의 index와 fitness 저장
		for (int i = 0; i < 30; i++) { 
			if (fitArr[i] < replacementFitness) { replacementIndex = i; replacementFitness = fitArr[i]; };
		}

		// offspring을 가장 worst한 population자리에 바꿔넣음
		copyArray(offspring, arr[replacementIndex]);
		fitArr[replacementIndex] = offspringFit;


//5. 30개의 population 중 best 찾아서 저장
		int bestIndex = 0; int bestFitness = fitArr[0];
		for (int i = 0; i < 30; i++) {
			if (fitArr[i] > bestFitness) { bestIndex = i; bestFitness = fitArr[i]; }
		}

		copyArray(arr[bestIndex], bestArr[k]);
		fitBestArr[k] = bestFitness;

		printf("Best: ");
		printArray(bestArr[k]);
		printf("  (f: %d)\n", fitBestArr[k]);

	}
} // main 함수 끝


void onePointCrossover(int parent1[ARRAY_SIZE], int parent2[ARRAY_SIZE], int crossoverOffspring[ARRAY_SIZE]) { // one-point crossover 실행 함수
	int cutPoint = rand() % (ARRAY_SIZE - 1) + 1; // 1부터 19 사이 cutPoint 무작위 생성
	for (int i = 0; i < cutPoint; i++) crossoverOffspring[i] = parent1[i];
	for (int i = cutPoint; i < ARRAY_SIZE; i++) crossoverOffspring[i] = parent2[i];
}

void bitflipMutation(int offspring[ARRAY_SIZE]) {
	int randomArr[ARRAY_SIZE] = { 0 };
	for (int i = 0; i < ARRAY_SIZE; i++) {
		randomArr[i] = rand() % 1000; // 각 배열의 index에 [0,999] 사이의 랜덤 숫자 주기
		if (randomArr[i] < 500) offspring[i] = 1 - offspring[i]; // Mutation Probability를 0.5로 설정, offspring의 값이 0이면 1로, 1이면 0으로 바꿔서 돌연변이 생성
	}
	
}

void copyArray(int originalArray[ARRAY_SIZE], int copyArray[ARRAY_SIZE]) { // 배열 복사 함수
	for (int i = 0; i < ARRAY_SIZE; i++) copyArray[i] = originalArray[i];
}

void printArray(int array[ARRAY_SIZE]) { // 배열 출력 함수
	for (int i = 0; i < ARRAY_SIZE; i++) printf("%d", array[i]);
}