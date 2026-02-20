"""
Notificaciones de Telegram para Bot de Trading Polymarket
Envía notificaciones de trades, P&L y alertas
"""

import os
import requests
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class TelegramNotifier:
    """Enviar notificaciones por Telegram"""

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)

        if self.enabled:
            print(f"[TelegramNotifier] ✅ Activado (Chat ID: {self.chat_id})")
        else:
            print(f"[TelegramNotifier] ⚠️  Desactivado (credenciales faltantes)")

    def send_message(self, message: str, silent: bool = False) -> bool:
        """Enviar mensaje a Telegram"""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_notification": silent
            }

            response = requests.post(url, data=data, timeout=10)

            if response.status_code == 200:
                return True
            else:
                print(f"[TelegramNotifier] ❌ Error: {response.status_code}")
                return False

        except Exception as e:
            print(f"[TelegramNotifier] ❌ Error: {e}")
            return False

    def notify_trade_buy(self, market: str, bucket: str, price: float,
                        shares: float, amount: float, reason: str,
                        balance: float, invested: float = 0, mode: str = "REAL",
                        strategy: str = "STANDARD"):
        """Notificar compra ejecutada"""
        emoji = "🔴" if mode == "REAL" else "📄"

        # Precio objetivo (95¢ normal, 99¢ moonshot)
        target_price = 0.95 if strategy != "MOONSHOT" else 0.99
        price_cents = price * 100
        target_cents = target_price * 100

        message = f"""
{emoji} <b>{mode} - COMPRA</b>
━━━━━━━━━━━━━━━━
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Entrada:</b> {price_cents:.0f}¢ → <b>Objetivo:</b> {target_cents:.0f}¢
<b>Tamaño:</b> ${amount:.2f} ({shares:.1f} shares)
<b>Razón:</b> {reason}
<b>Estrategia:</b> {strategy}
━━━━━━━━━━━━━━━━
💰 <b>Cash:</b> ${balance:.2f}
📊 <b>Invertido:</b> ${invested:.2f}
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_trade_sell(self, market: str, bucket: str, price: float,
                         shares: float, pnl: float, pnl_pct: float,
                         balance: float, reason: str = "", mode: str = "REAL"):
        """Notificar venta ejecutada"""
        emoji = "🔴" if mode == "REAL" else "📄"
        pnl_emoji = "💰" if pnl > 0 else "📉"
        pnl_sign = "+" if pnl >= 0 else ""
        price_cents = price * 100

        message = f"""
{emoji} <b>{mode} - VENTA</b>
━━━━━━━━━━━━━━━━
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Salida:</b> {price_cents:.0f}¢ ({shares:.1f} shares)
{f"<b>Razón:</b> {reason}" if reason else ""}

{pnl_emoji} <b>P&L:</b> {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)
━━━━━━━━━━━━━━━━
💰 <b>Balance:</b> ${balance:.2f}
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_daily_loss_warning(self, daily_pnl: float, limit: float):
        """Notificar que se acerca el límite de pérdida diaria"""
        message = f"""
⚠️ <b>AVISO PÉRDIDA DIARIA</b>
━━━━━━━━━━━━━━━━
<b>Pérdida actual:</b> -${abs(daily_pnl):.2f}
<b>Límite:</b> ${limit:.2f}
<b>Restante:</b> ${limit - abs(daily_pnl):.2f}

🛑 Considera parar el trading por hoy
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_daily_loss_limit_hit(self, daily_pnl: float, limit: float):
        """Notificar que se alcanzó el límite de pérdida diaria"""
        message = f"""
🛑 <b>LÍMITE PÉRDIDA ALCANZADO</b>
━━━━━━━━━━━━━━━━
<b>Pérdida total:</b> -${abs(daily_pnl):.2f}
<b>Límite:</b> ${limit:.2f}

⚠️ <b>TRADING PAUSADO POR HOY</b>
¡Intervención manual requerida!
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_error(self, error: str, context: str = ""):
        """Notificar error crítico"""
        message = f"""
❌ <b>ERROR</b>
━━━━━━━━━━━━━━━━
<b>Error:</b> {error}
{f"<b>Contexto:</b> {context}" if context else ""}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_startup(self, mode: str, balance: float, positions: int):
        """Notificar inicio del bot"""
        emoji = "🔴" if mode == "REAL" else "📄"

        message = f"""
🚀 <b>BOT INICIADO</b>
━━━━━━━━━━━━━━━━
<b>Modo:</b> {emoji} {mode}
<b>Balance:</b> ${balance:.2f}
<b>Posiciones abiertas:</b> {positions}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message.strip(), silent=True)

    def notify_daily_summary(self, trades: int, wins: int, losses: int,
                            total_pnl: float, balance: float, mode: str):
        """Notificar resumen diario"""
        emoji = "🔴" if mode == "REAL" else "📄"
        win_rate = (wins / trades * 100) if trades > 0 else 0
        pnl_emoji = "💰" if total_pnl > 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""

        message = f"""
📊 <b>RESUMEN DIARIO - {mode}</b>
━━━━━━━━━━━━━━━━
<b>Trades:</b> {trades} ({wins}W / {losses}L)
<b>Win Rate:</b> {win_rate:.1f}%

{pnl_emoji} <b>P&L Total:</b> {pnl_sign}${total_pnl:.2f}

💰 <b>Balance Final:</b> ${balance:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_position_update(self, market: str, bucket: str, entry: float,
                              current: float, unrealized_pnl: float, mode: str):
        """Notificar actualización de posición"""
        emoji = "🔴" if mode == "REAL" else "📄"
        pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
        pnl_sign = "+" if unrealized_pnl >= 0 else ""
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0

        message = f"""
{emoji} <b>ACTUALIZACIÓN POSICIÓN</b>
━━━━━━━━━━━━━━━━
<b>Mercado:</b> {market}
<b>Bucket:</b> {bucket}
<b>Entrada:</b> ${entry:.3f}
<b>Actual:</b> ${current:.3f}

{pnl_emoji} <b>P&L No Realizado:</b> {pnl_sign}${unrealized_pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip(), silent=True)

    def notify_positions_summary(self, positions: list, balance: float, mode: str):
        """Enviar resumen completo de posiciones (cada 2 horas)"""
        emoji = "🔴" if mode == "REAL" else "📄"

        if not positions:
            message = f"""
📊 <b>PORTFOLIO {mode}</b>
━━━━━━━━━━━━━━━━
💰 <b>Cash:</b> ${balance:.2f}
📂 <b>Posiciones abiertas:</b> 0

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            self.send_message(message.strip(), silent=True)
            return

        # Calcular totales
        total_invested = sum(p.get('size', 0) * p.get('avg_entry_price', 0) for p in positions)
        total_unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
        total_equity = balance + total_invested + total_unrealized_pnl

        # Construir tabla compacta
        positions_table = "<code>"
        positions_table += "EVENTO      BUCKET  AVG NOW P&L\n"
        positions_table += "──────────────────────────\n"

        # Construir filas
        for pos in positions:
            # Event label (e.g., "Feb13-Feb20")
            event = pos.get('event_slug', '')[:11]  # Truncate to fit
            bucket = pos.get('range_label', 'N/A')
            entry = pos.get('avg_entry_price', 0) * 100  # a centavos
            current = pos.get('current_price', 0) * 100  # a centavos
            pnl = pos.get('unrealized_pnl', 0)
            pnl_sign = "+" if pnl >= 0 else ""

            positions_table += f"{event:<11} {bucket:<7} {entry:>2.0f}¢ {current:>2.0f}¢ {pnl_sign}{pnl:>4.2f}\n"

        positions_table += "</code>"

        # Construir mensaje completo
        pnl_sign = "+" if total_unrealized_pnl >= 0 else ""
        pnl_emoji = "📈" if total_unrealized_pnl >= 0 else "📉"

        message = f"""
📊 <b>PORTFOLIO {mode}</b>
━━━━━━━━━━━━━━━━
{positions_table}
━━━━━━━━━━━━━━━━
💰 <b>Cash:</b> ${balance:.2f}
📊 <b>Invertido:</b> ${total_invested:.2f}
💼 <b>Equity:</b> ${total_equity:.2f}
{pnl_emoji} <b>P&L Total:</b> {pnl_sign}${total_unrealized_pnl:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message.strip(), silent=True)

    def notify_stop_loss_triggered(self, market: str, bucket: str, entry: float,
                                   exit_price: float, pnl: float, reason: str = ""):
        """Alerta cuando se dispara stop loss"""
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_pct = ((exit_price - entry) / entry * 100) if entry > 0 else 0

        message = f"""
🛑 <b>STOP LOSS ACTIVADO</b>
━━━━━━━━━━━━━━━━
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Entrada:</b> {entry*100:.0f}¢ → <b>Salida:</b> {exit_price*100:.0f}¢
{f"<b>Razón:</b> {reason}" if reason else ""}

📉 <b>Pérdida:</b> {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_victory_lap(self, market: str, bucket: str, entry: float,
                          exit_price: float, pnl: float, pnl_pct: float):
        """Alerta de victoria (>90% profit)"""
        message = f"""
🎉 <b>¡VICTORIA!</b>
━━━━━━━━━━━━━━━━
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Entrada:</b> {entry*100:.0f}¢ → <b>Salida:</b> {exit_price*100:.0f}¢

💰 <b>Ganancia:</b> +${pnl:.2f} (+{pnl_pct:.1f}%)

🚀 ¡Gran victoria asegurada!
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_proximity_danger(self, market: str, bucket: str, tweets_to_max: int,
                               current_count: int, bucket_max: int):
        """Alerta cuando se acerca al máximo del bucket"""
        message = f"""
⚠️ <b>PELIGRO DE PROXIMIDAD</b>
━━━━━━━━━━━━━━━━
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Actual:</b> {current_count} tweets
<b>Máximo bucket:</b> {bucket_max}
<b>Distancia:</b> {tweets_to_max} tweets

🎯 Preparando salida pronto
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_moonshot_alert(self, market: str, bucket: str, entry: float,
                             current: float, pnl: float, pnl_pct: float):
        """Alerta cuando moonshot gana >50%"""
        message = f"""
🌙 <b>¡MOONSHOT SUBIENDO!</b>
━━━━━━━━━━━━━━━━
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Entrada:</b> {entry*100:.0f}¢ → <b>Ahora:</b> {current*100:.0f}¢

📈 <b>No realizado:</b> +${pnl:.2f} (+{pnl_pct:.0f}%)

🚀 ¡Moonshot despegando!
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_big_win(self, market: str, bucket: str, entry: float,
                      current: float, pnl: float, pnl_pct: float):
        """Alerta cuando posición gana >20%"""
        message = f"""
💎 <b>GRAN GANANCIA</b>
━━━━━━━━━━━━━━━━
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Entrada:</b> {entry*100:.0f}¢ → <b>Ahora:</b> {current*100:.0f}¢

📈 <b>No realizado:</b> +${pnl:.2f} (+{pnl_pct:.1f}%)

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip(), silent=True)

    def notify_drawdown_alert(self, daily_pnl: float, high_water_mark: float):
        """Alerta de drawdown significativo"""
        drawdown_pct = (daily_pnl / high_water_mark * 100) if high_water_mark > 0 else 0

        message = f"""
📉 <b>ALERTA DRAWDOWN</b>
━━━━━━━━━━━━━━━━
<b>P&L Actual:</b> ${daily_pnl:.2f}
<b>Desde máximo:</b> {drawdown_pct:.1f}%

⚠️ Considera revisar posiciones abiertas
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_order_failed(self, market: str, bucket: str, side: str,
                           price: float, reason: str = ""):
        """Alerta cuando falla ejecución de orden"""
        message = f"""
❌ <b>ORDEN FALLIDA</b>
━━━━━━━━━━━━━━━━
<b>Lado:</b> {side}
<b>Evento:</b> {market[:30]}
<b>Bucket:</b> {bucket}
<b>Precio:</b> {price*100:.0f}¢
{f"<b>Razón:</b> {reason}" if reason else ""}

⚠️ FOK rechazada - sin liquidez
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip(), silent=True)

    def notify_low_balance(self, balance: float, threshold: float):
        """Alerta de balance bajo"""
        message = f"""
⚠️ <b>AVISO BALANCE BAJO</b>
━━━━━━━━━━━━━━━━
<b>Balance actual:</b> ${balance:.2f}
<b>Umbral:</b> ${threshold:.2f}

💰 Considera depositar más fondos
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        self.send_message(message.strip())

    def notify_daily_digest(self, date: str, trades_buy: int, trades_sell: int,
                           wins: int, losses: int, total_pnl: float,
                           best_trade: float, worst_trade: float,
                           open_positions: int, balance: float, mode: str):
        """Enviar resumen diario a medianoche"""
        emoji = "🔴" if mode == "REAL" else "📄"
        total_trades = trades_sell  # Solo trades cerrados
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        pnl_emoji = "💰" if total_pnl > 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""

        message = f"""
📅 <b>RESUMEN DEL DÍA - {date}</b>
━━━━━━━━━━━━━━━━
{emoji} <b>Modo:</b> {mode}

📊 <b>TRADES</b>
• Abiertos: {trades_buy}
• Cerrados: {trades_sell}
• Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)

{pnl_emoji} <b>RENDIMIENTO</b>
• P&L Total: {pnl_sign}${total_pnl:.2f}
• Mejor Trade: +${best_trade:.2f}
• Peor Trade: -${abs(worst_trade):.2f}

💼 <b>ESTADO</b>
• Posiciones abiertas: {open_positions}
• Balance: ${balance:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message.strip())

    def test_connection(self) -> bool:
        """Probar conexión de Telegram"""
        if not self.enabled:
            print("[TelegramNotifier] ❌ No activado")
            return False

        message = "🧪 <b>Mensaje de Prueba</b>\n\n¡Las notificaciones de Telegram funcionan! ✅"
        success = self.send_message(message)

        if success:
            print("[TelegramNotifier] ✅ Mensaje de prueba enviado correctamente")
        else:
            print("[TelegramNotifier] ❌ Fallo al enviar mensaje de prueba")

        return success


# Script de prueba
if __name__ == "__main__":
    notifier = TelegramNotifier()

    if notifier.enabled:
        print("\n🧪 Probando notificaciones de Telegram...\n")

        # Probar conexión
        if notifier.test_connection():
            print("\n✅ ¡Notificaciones de Telegram configuradas correctamente!")
        else:
            print("\n❌ Fallo al enviar mensaje de prueba. Verifica tus credenciales.")
    else:
        print("\n⚠️  Notificaciones de Telegram no configuradas.")
        print("Añade TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID al archivo .env")
