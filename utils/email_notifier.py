"""
Email notifier for medication scheduler
- Sends reminder emails at scheduled times
- Sends immediate confirmation email when schedule is created (to profile email)
- Provides SMTP status diagnostics
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def _get_smtp_config():
    smtp_email = os.getenv("SMTP_EMAIL", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com").strip()
    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except:
        smtp_port = 587
    return smtp_email, smtp_password, smtp_server, smtp_port

def get_smtp_status():
    """Returns dict explaining SMTP configuration status"""
    smtp_email, smtp_password, smtp_server, smtp_port = _get_smtp_config()
    is_configured = bool(smtp_email and smtp_password and smtp_email != "your_email@gmail.com" and smtp_password != "your_app_password")

    issues = []
    if not smtp_email:
        issues.append("SMTP_EMAIL not set in .env")
    elif smtp_email == "your_email@gmail.com":
        issues.append("SMTP_EMAIL is still placeholder 'your_email@gmail.com' - set your real Gmail")

    if not smtp_password:
        issues.append("SMTP_PASSWORD not set in .env")
    elif smtp_password == "your_app_password":
        issues.append("SMTP_PASSWORD is still placeholder - set your Gmail App Password")

    if not smtp_server:
        issues.append("SMTP_SERVER not set")

    return {
        "is_configured": is_configured,
        "smtp_email": smtp_email or "(not set)",
        "smtp_server": smtp_server,
        "smtp_port": smtp_port,
        "has_password": bool(smtp_password) and smtp_password != "your_app_password",
        "issues": issues,
        "mode": "REAL SMTP" if is_configured else "MOCK (logged to email_log.txt)",
        "instructions": [
            "1. For Gmail: Enable 2FA, then create App Password at https://myaccount.google.com/apppasswords",
            "2. Set in .env: SMTP_EMAIL=your_real_gmail@gmail.com",
            "3. SMTP_PASSWORD=your 16-char app password (no spaces)",
            "4. SMTP_SERVER=smtp.gmail.com, SMTP_PORT=587",
            "5. Restart app: python app.py",
            "6. Test: POST /api/test-email"
        ] if not is_configured else [
            f"SMTP configured to send from {smtp_email} via {smtp_server}:{smtp_port}",
            "Emails will be sent to user's profile email (the email in profile, not SMTP_EMAIL)",
            "SMTP_EMAIL is the SENDER, profile email is the RECIPIENT"
        ]
    }

def _log_mock_email(to_email: str, subject: str, html_content: str, reason="SMTP not configured"):
    print(f"[MOCK EMAIL - {reason}] Would send to {to_email}: {subject}")
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "email_log.txt")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now()}] MOCK REASON: {reason}\nTO:{to_email} | SUBJ:{subject}\n{html_content[:5000]}\n{'='*80}\n")
        print(f"📁 Mock email logged to {log_path}")
    except Exception as e:
        print(f"Log write error: {e}")

def _send_email(to_email: str, subject: str, html_content: str):
    """
    Sends email, returns dict {success, is_mock, message, error}
    """
    smtp_email, smtp_password, smtp_server, smtp_port = _get_smtp_config()
    status = get_smtp_status()

    if not status["is_configured"]:
        reason = "; ".join(status["issues"]) if status["issues"] else "SMTP not configured"
        _log_mock_email(to_email, subject, html_content, reason=reason)
        return {
            "success": True,  # Considered success in mock mode for demo
            "is_mock": True,
            "real_sent": False,
            "message": f"MOCK mode: Email logged to email_log.txt, not actually sent. Reason: {reason}. Configure SMTP in .env to send real emails to {to_email}",
            "to": to_email,
            "subject": subject,
            "smtp_status": status
        }

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = smtp_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_content, "html"))

        print(f"📤 Attempting SMTP send: From {smtp_email} To {to_email} via {smtp_server}:{smtp_port}")
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=20)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, to_email, msg.as_string())
        server.quit()
        print(f"✅ REAL Email sent to {to_email}: {subject}")
        return {
            "success": True,
            "is_mock": False,
            "real_sent": True,
            "message": f"Email successfully sent via SMTP to {to_email}",
            "to": to_email,
            "subject": subject,
            "smtp_status": status
        }
    except smtplib.SMTPAuthenticationError as e:
        err = f"SMTP Authentication failed: {e}. Check SMTP_EMAIL and SMTP_PASSWORD (use Gmail App Password, not regular password)"
        print(f"❌ {err}")
        _log_mock_email(to_email, f"[AUTH FAILED - logged] {subject}", html_content, reason=err)
        return {
            "success": False,
            "is_mock": False,
            "real_sent": False,
            "message": err,
            "error": str(e),
            "to": to_email,
            "subject": subject,
            "smtp_status": status
        }
    except Exception as e:
        err = f"SMTP send failed: {e}"
        print(f"❌ {err}")
        _log_mock_email(to_email, f"[FAILED - logged] {subject}", html_content, reason=err)
        return {
            "success": False,
            "is_mock": False,
            "real_sent": False,
            "message": err,
            "error": str(e),
            "to": to_email,
            "subject": subject,
            "smtp_status": status
        }

def send_medication_email(to_email: str, medicine_card: dict, user_name: str = "User"):
    """Reminder email sent at scheduled time - returns dict status"""
    subject = f"💊 Care Companion Reminder: Time to take {medicine_card.get('name', 'your medicine')}"

    html_content = f"""
    <html>
    <body style="font-family: 'Segoe UI', Arial; background:#e8f8f3; padding:20px; margin:0;">
        <div style="max-width:600px; margin:auto; background:white; border-radius:16px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <div style="background: linear-gradient(135deg, #0a3d2e, #16a085); padding:24px; color:white;">
                <h2 style="margin:0; font-weight:700; font-size:20px;">Care Companion</h2>
                <p style="margin:4px 0 0; opacity:0.9; font-size:13px;">Medication Reminder ⏰</p>
            </div>
            <div style="padding:24px;">
                <p style="margin:0 0 8px; font-size:14px;">Hi <strong>{user_name}</strong>,</p>
                <p style="margin:0 0 16px; font-size:14px; color:#334e46;">This is a gentle reminder to take your medicine now:</p>
                <div style="background:#f0faf7; border:1px solid #b2dfd1; border-radius:12px; padding:16px; margin:16px 0;">
                    <h3 style="margin:0 0 12px; color:#0a3d2e; font-size:16px;">💊 {medicine_card.get('name','Medicine')}</h3>
                    <table style="width:100%; font-size:13px; border-collapse:separate; border-spacing:0 6px;">
                        <tr><td style="color:#5b7c72; width:90px;"><strong>Dosage:</strong></td><td>{medicine_card.get('dosage','-')}</td></tr>
                        <tr><td style="color:#5b7c72;"><strong>Frequency:</strong></td><td>{medicine_card.get('frequency','-')}</td></tr>
                        <tr><td style="color:#5b7c72;"><strong>Duration:</strong></td><td>{medicine_card.get('duration','-')}</td></tr>
                        <tr><td style="color:#5b7c72;"><strong>Instructions:</strong></td><td>{medicine_card.get('instructions','Take as prescribed')}</td></tr>
                    </table>
                    <div style="margin-top:12px; background:#fff3cd; border:1px solid #ffe082; padding:10px 12px; border-radius:8px; font-size:12px; line-height:1.5;">
                        <strong>⚠️ Precautions:</strong> {medicine_card.get('precautions','Follow doctor advice, check expiry, keep away from children.')}
                    </div>
                </div>
                <div style="background:#fff8e1; border-left:4px solid #ffb300; padding:12px 14px; border-radius:0 8px 8px 0; font-size:12px; line-height:1.5; color:#6d4c00;">
                    <strong style="color:#0a3d2e;">Safety Guardrails:</strong><br>
                    • Take exactly as prescribed<br>
                    • Do not double dose if missed without doctor advice<br>
                    • Watch for side effects: rash, swelling, breathing difficulty → seek help<br>
                    • Store properly, check expiry<br>
                    • This is a reminder tool, not medical advice.
                </div>
                <p style="font-size:11px; color:#888; margin-top:20px; line-height:1.4;">You received this because you scheduled it in Care Companion.<br>To stop: remove from scheduler in app.<br>Profile email: {to_email}</p>
            </div>
            <div style="background:#0a3d2e; color:#a7f3d0; text-align:center; padding:12px; font-size:10px;">
                Care Companion • AI health info, not a medical professional • For emergency call 112
            </div>
        </div>
    </body>
    </html>
    """
    result = _send_email(to_email, subject, html_content)
    # For backward compatibility, also return bool in some contexts, but we return dict now
    # Callers should check dict['success']
    return result

def send_schedule_confirmation_email(to_email: str, medicine_card: dict, schedule_info: dict, user_name: str = "User"):
    """Immediate email sent WHEN schedule is created, to profile email"""
    times = schedule_info.get('times', [])
    times_str = ", ".join(times) if times else "Not set"
    start_date = schedule_info.get('start_date', 'Today')
    end_date = schedule_info.get('end_date', 'Until removed')

    subject = f"✅ Care Companion: Medication Schedule Created - {medicine_card.get('name', 'your medicine')}"

    html_content = f"""
    <html>
    <body style="font-family: 'Segoe UI', Arial; background:#e8f8f3; padding:20px; margin:0;">
        <div style="max-width:600px; margin:auto; background:white; border-radius:16px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.08);">
            <div style="background: linear-gradient(135deg, #0a3d2e, #0f5d44); padding:24px; color:white;">
                <h2 style="margin:0; font-weight:700; font-size:20px;">Care Companion</h2>
                <p style="margin:4px 0 0; opacity:0.9; font-size:13px;">✅ Medication Schedule Confirmation</p>
            </div>
            <div style="padding:24px;">
                <p style="margin:0 0 8px; font-size:14px;">Hi <strong>{user_name}</strong>,</p>
                <p style="margin:0 0 16px; font-size:14px; color:#334e46; line-height:1.5;">
                    Your medication schedule has been <strong>successfully created</strong>. 
                    You will now receive email reminders at the times you selected.
                </p>
                
                <div style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:12px; padding:14px 16px; margin:0 0 16px 0; font-size:13px;">
                    <strong style="color:#065f46;">⏰ Your Schedule:</strong><br>
                    <span style="font-size:15px; font-weight:700; color:#0a3d2e;">{times_str}</span> (24h format)<br>
                    <span style="color:#5b7c72; font-size:12px;">Start: {start_date} | End: {end_date if end_date else 'Ongoing'}</span><br>
                    <span style="color:#5b7c72; font-size:12px;">Reminders will be sent to: <strong>{to_email}</strong> (your profile email - from your Care Companion profile)</span>
                </div>

                <div style="background:#f0faf7; border:1px solid #b2dfd1; border-radius:12px; padding:16px; margin:0 0 16px 0;">
                    <h3 style="margin:0 0 12px; color:#0a3d2e; font-size:16px;">💊 {medicine_card.get('name','Medicine')}</h3>
                    <table style="width:100%; font-size:13px; border-collapse:separate; border-spacing:0 6px;">
                        <tr><td style="color:#5b7c72; width:90px;"><strong>Dosage:</strong></td><td>{medicine_card.get('dosage','-')}</td></tr>
                        <tr><td style="color:#5b7c72;"><strong>Frequency:</strong></td><td>{medicine_card.get('frequency','-')}</td></tr>
                        <tr><td style="color:#5b7c72;"><strong>Duration:</strong></td><td>{medicine_card.get('duration','-')}</td></tr>
                        <tr><td style="color:#5b7c72;"><strong>Instructions:</strong></td><td>{medicine_card.get('instructions','Take as prescribed')}</td></tr>
                    </table>
                    <div style="margin-top:12px; background:#fff3cd; border:1px solid #ffe082; padding:10px 12px; border-radius:8px; font-size:12px; line-height:1.5;">
                        <strong>⚠️ Precautions:</strong> {medicine_card.get('precautions','Follow doctor advice, check expiry, keep away from children.')}
                    </div>
                </div>

                <div style="background:#fff8e1; border-left:4px solid #ffb300; padding:12px 14px; border-radius:0 8px 8px 0; font-size:12px; line-height:1.5; color:#6d4c00; margin-bottom:16px;">
                    <strong style="color:#0a3d2e;">Safety Guardrails:</strong><br>
                    • Take exactly as prescribed by your doctor<br>
                    • Do not double dose if you miss a dose<br>
                    • Watch for allergic reactions: rash, swelling, breathing difficulty → emergency<br>
                    • Check expiry, store as per label, keep away from children<br>
                    • This app provides information only, not medical advice
                </div>

                <div style="background:#f6fffd; border:1px dashed #b2dfd1; border-radius:10px; padding:12px; font-size:12px; color:#4a756a;">
                    <strong>What happens next?</strong><br>
                    • You will receive a reminder email at <strong>{times_str}</strong> every day<br>
                    • Each reminder contains this medication card<br>
                    • To stop: Care Companion → Medications → Remove<br>
                    • Your profile email <strong>{to_email}</strong> is used for all reminders (this is the email from your profile, not the sender)
                </div>
            </div>
            <div style="background:#0a3d2e; color:#a7f3d0; text-align:center; padding:12px; font-size:10px;">
                Care Companion • For emergency call 112 • Not medical advice
            </div>
        </div>
    </body>
    </html>
    """
    return _send_email(to_email, subject, html_content)
