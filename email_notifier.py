import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_trade_email(subject: str, message: str,
                     sender_email: str, sender_password: str,
                     receiver_email: str,
                     smtp_server: str = "smtp.gmail.com",
                     smtp_port: int = 587):
    try:
        # 이메일 헤더 설정
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # 본문 추가
        msg.attach(MIMEText(message, 'plain'))

        # SMTP 세션 생성 및 로그인
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        # 이메일 전송
        server.send_message(msg)
        server.quit()
        print(f"📧 이메일 전송 완료: {receiver_email}")

    except Exception as e:
        print("❌ 이메일 전송 실패:", e)


# ✅ 예시 사용법 (자동매매 코드에서 호출)
if __name__ == "__main__":
    sender = "your.email@gmail.com"              # 발신자 이메일
    app_pw = "your_app_password_here"            # 앱 비밀번호 (2단계 인증 시)
    recipient = "your.email@gmail.com"           # 수신자 이메일

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[자동매매 알림] 진입 발생 - {now}"
    content = (
        f"진입 시간: {now}\n"
        f"진입 방향: 📈 LONG\n"
        f"진입가: 0.012831\n"
        f"익절가: 0.012899\n"
        f"예측 신뢰도: 82.5%\n"
        f"예상 순익: $1.05\n"
        f"잔고: $581.20"
    )

    send_trade_email(subject, content, sender, app_pw, recipient)