
def solution1(n):
    answer = 0

    while n>0:
        if n%5==0:
            answer += n //5
            break
        n -= 3
        answer += 1
    else:
        answer = -1
    return answer


def solution2(n, r, c):
    # 대각선 합(1-based): s = r + c
    s = r + c

    # 이 대각선의 길이
    if s <= n + 1:
        length = s - 1
        count_before = (s - 2) * (s - 1) // 2  # 이전 대각선까지의 총 개수(삼각수)
        row_start = 1
        col_start = s - 1
    else:
        length = 2 * n + 1 - s
        t = s - (n + 1)  # 감소 구간에서 몇 번째 대각선인지(1..n-1)
        count_before = n * (n + 1) // 2 + (t - 1) * n - (t - 1) * t // 2
        row_start = s - n
        col_start = n

    # 해당 대각선 내에서의 0-based 인덱스 (위→아래로 내려가며)
    k = r - row_start

    # 지그재그 방향: s가 홀수면 위→아래로 증가, 짝수면 반대로
    if s % 2 == 1:  # 홀수: top->bottom 증가
        answer= count_before + k + 1
    else:           # 짝수: top->bottom 감소
        answer= count_before + (length - k)

    return answer


def solution3(p):
    if not p :
        return 0
    n = len(p)
    count = max(Counter(p).values())
    answer = n -count
    return answer



def diag_math22(n, r, c):
    s = r + c
    if s <= n + 1:
        start = (s - 2) * (s - 1) // 2 + 1
        if s % 2 == 1:
            return start + (r - 1)
        else:
            return start + (s - 1 - (r - 1))
    else:
        t = s - n
        start = n*(n+1)//2 + (t-1)*n - (t-1)*t//2 + 1
        length = 2*n + 1 - s
        if s % 2 == 1:
            return start + (r - (s - n))
        else:
            return start + (length - (r - (s - n)) - 1)
def solution_test(n, r, c):
    s = r + c
    if s <= n + 1:  # ↑ 길이 1..n
        start  = (s - 2) * (s - 1) // 2 + 1
        length = s - 1
        k_in   = r - 1                  # 대각선 내 위→아래 0-based
        if s % 2 == 1:                  # 홀수합: 위→아래 증가
            return start + k_in
        else:                           # 짝수합: 아래→위 증가
            return start + (length - 1 - k_in)
    else:              # ↓ 길이 n-1..1
        Tn = n * (n + 1) // 2           # s = n+1 까지 누적
        k  = s - n - 2                  # s 전까지 지나간 '감소' 대각선 개수
        if k < 0: k = 0
        sum_dec = k * (2*(n-1) - (k-1)) // 2   # (n-1)+(n-2)+... 합
        start   = Tn + sum_dec + 1

        length = 2 * n + 1 - s
        top_r  = s - n
        k_in   = r - top_r
        if s % 2 == 1:                  # 홀수합: 위→아래 증가
            return start + k_in
        else:                           # 짝수합: 아래→위 증
            return start + (length - 1 - k_in)

from collections import Counter



if __name__ == "__main__":
    print(diag_value(6,5,4))