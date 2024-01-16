// 3주차 Locus-based 
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define ARRAY_SIZE 10

void OrderLocusTransfer(const int order[], int locus[]);

int main() {
	// 배열 정의 (길이 10)
	int order[ARRAY_SIZE];

	// 사용자로부터 order-based[] 입력받기
	printf("Enter the first parent order-based array (10 binary digits, ex)2581470963 ): ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		scanf("%1d", &order[i]);
	}

	// locus-based 수행
	int locus[10];
	OrderLocusTransfer(order, locus);

	// 결과 출력
	printf("\nOrder-based: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d ", order[i]);
	}
	printf("\n");

	printf("Locus-based: ");
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%d ", locus[i]);
	}
	printf("\n");

	return 0;
}

// OrderLocusTransfer 함수 구현
void OrderLocusTransfer(const int order[], int locus[]) {
	for (int i = 0; i < ARRAY_SIZE; i++) { // index 순서대로 찾기 위한 i
		for (int k = 0; k < ARRAY_SIZE; k++) { // 해당 순서의 index와 일치하는 것을 찾기 위한 for문 k
			if (order[k] == i) {
				locus[i] = order[k+1];
				if (k + 1 == ARRAY_SIZE) {
					locus[i] = order[0];
				}
			}
		}
	}
}