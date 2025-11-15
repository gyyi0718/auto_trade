# debug_balance.py
# 잔고 조회 디버깅 스크립트
import json
import hmac
import hashlib
import time
import requests

# API 키 로드
with open('api_keys.json', 'r') as f:
    api_keys = json.load(f)

api_key = api_keys['api_key']
api_secret = api_keys['api_secret']
testnet = api_keys.get('testnet', True)

base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
recv_window = "5000"

print(f"\n🔍 잔고 조회 디버깅")
print(f"네트워크: {'테스트넷' if testnet else '메인넷 🔴'}")
print("="*60)

# 서명 생성
def generate_signature(timestamp, params_str):
    sign_str = timestamp + api_key + recv_window + params_str
    return hmac.new(
        api_secret.encode('utf-8'),
        sign_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

# 잔고 조회
endpoint = "/v5/account/wallet-balance"
timestamp = str(int(time.time() * 1000))
params = {"accountType": "UNIFIED"}
params_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])

signature = generate_signature(timestamp, params_str)

headers = {
    "X-BAPI-API-KEY": api_key,
    "X-BAPI-TIMESTAMP": timestamp,
    "X-BAPI-SIGN": signature,
    "X-BAPI-RECV-WINDOW": recv_window,
    "Content-Type": "application/json"
}

url = f"{base_url}{endpoint}"

print(f"\n📡 요청:")
print(f"   URL: {url}")
print(f"   Params: {params}")

try:
    response = requests.get(url, params=params, headers=headers, timeout=10)
    result = response.json()
    
    print(f"\n📥 응답:")
    print(f"   Status: {response.status_code}")
    print(f"   RetCode: {result.get('retCode')}")
    print(f"   RetMsg: {result.get('retMsg')}")
    
    if result.get('retCode') == 0:
        print(f"\n✅ 성공!")
        
        # 전체 응답 출력 (구조 확인용)
        print(f"\n📋 전체 응답:")
        print(json.dumps(result, indent=2))
        
        # USDT 찾기
        print(f"\n💰 USDT 잔고 찾기:")
        coin_list = result.get('result', {}).get('list', [])
        print(f"   계좌 개수: {len(coin_list)}")
        
        for idx, account in enumerate(coin_list):
            print(f"\n   계좌 #{idx + 1}:")
            print(f"   - accountType: {account.get('accountType', 'N/A')}")
            
            coins = account.get('coin', [])
            print(f"   - 코인 개수: {len(coins)}")
            
            for coin_info in coins:
                coin_name = coin_info.get('coin', 'Unknown')
                if coin_name == 'USDT':
                    print(f"\n   🎯 USDT 발견!")
                    print(f"      - walletBalance: {coin_info.get('walletBalance')}")
                    print(f"      - availableToWithdraw: {coin_info.get('availableToWithdraw')}")
                    print(f"      - equity: {coin_info.get('equity')}")
                    print(f"      - availableToBorrow: {coin_info.get('availableToBorrow')}")
                    
                    # 실제 사용 가능 금액
                    available = coin_info.get('availableToWithdraw', '0')
                    print(f"\n      💵 사용 가능: {available}")
                    
                    if available == '' or available is None:
                        print(f"      ⚠️ 빈 값! 기본값 0 사용")
                        available = '0'
                    
                    try:
                        balance = float(available)
                        print(f"      ✅ Float 변환: ${balance:,.2f}")
                    except ValueError as e:
                        print(f"      ❌ Float 변환 실패: {e}")
    else:
        print(f"\n❌ 실패!")
        print(f"   에러: {result.get('retMsg')}")
        
        # API 권한 체크
        if result.get('retCode') == 10003:
            print(f"\n⚠️ API 키 인증 실패!")
            print(f"   1. API 키가 올바른지 확인")
            print(f"   2. testnet 설정 확인 (testnet 키는 testnet: true)")
            print(f"   3. API 권한 확인 (Read-Write 필요)")

except Exception as e:
    print(f"\n❌ 예외 발생:")
    print(f"   {e}")

print(f"\n" + "="*60)
