import os, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASSWORD", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

def trimite_email_reset(email_dest: str, token: str):
    link = f"{FRONTEND_URL}/reset-password?token={token}"
    body = f"""
    <html><body>
      <h2>Resetare parolă Aceso</h2>
      <p>Apasă butonul de mai jos pentru a-ți reseta parola:</p>
      <a href="{link}" style="background:#6366f1;color:white;padding:12px 24px;
         border-radius:6px;text-decoration:none;display:inline-block;">
        Resetează parola
      </a>
      <p>Link-ul expiră în <strong>1 oră</strong>.</p>
      <p>Dacă nu tu ai solicitat resetarea, ignoră acest email.</p>
    </body></html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Resetare parolă Aceso"
    msg["From"]    = SMTP_USER
    msg["To"]      = email_dest
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, email_dest, msg.as_string())