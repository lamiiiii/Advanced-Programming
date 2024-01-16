// Locus-based 연습

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define ARRAY_SIZE 10

void orderLocusTransfer(int orderBased[ARRAY_SIZE], int locusBased[ARRAY_SIZE]);
void printArray(int array[10]);

int main() {
	// 배열 2개 생성
	int orderBased[ARRAY_SIZE];
	int locusBased[ARRAY_SIZE];

	// 사용자로부터 order-based 배열 입력받기
	printf("길이 10의 orderbased 배열을 입력하세요.(ex. 8467915023)\n");
	printf("Order-based: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		scanf("%1d", &orderBased[i]);
	}
	printf("\n");

	// 함수 실행
	orderLocusTransfer(orderBased, locusBased);

	// 출력
	printf("\nOrder-based: ");
	printArray(orderBased);
	printf("\nLocus-based: ");
	printArray(locusBased);

	return 0;
}

// order-based 배열을 locus-based 배열로 바꾸는 함수
void orderLocusTransfer(int orderBased[ARRAY_SIZE], int locusBased[ARRAY_SIZE]) {
	for (int i = 0; i < ARRAY_SIZE; i++) {
		for (int k = 0; k < ARRAY_SIZE; k++) {
			if (orderBased[k] == i) {
				locusBased[i] = orderBased[k + 1];
				if (k+1 == ARRAY_SIZE) {
					locusBased[i] = orderBased[0];
				}
			}
		}
	}
}

// 배열 출력 함수
void printArray(int array[10]) {
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d", array[i]);
	}
};