import os
import json
import time
import re
from datetime import datetime
from config import *

try:
    import database as db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

class PaperTrader:
    def __init__(self, initial_cash=PORTFOLIO_INITIAL_CASH):
        self.file_path = os.path.join(LOGS_DIR, PORTFOLIO_PAPER_TRADER)
        self.log_path = os.path.join(LOGS_DIR, TRADE_LOG)
        self.risk_pct_normal = RISK_PCT_NORMAL
        self.risk_pct_lotto = RISK_PCT_LOTTO
        self.risk_pct_moonshot = RISK_PCT_MOONSHOT
        self.max_moonshot_bet = MAX_MOONSHOT_BET
        self.min_bet = PORTFOLIO_MIN_BET
        self.portfolio = self._load()
        self._ensure_log_header()
        if not self.portfolio: self.portfolio = {"cash": initial_cash, "positions": {}, "history": []}

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f: return json.load(f)
            except: return None
        return None

    def _save(self):
        # 1. Escribir a archivo (OBLIGATORIO - fuente de verdad)
        with open(self.file_path, 'w') as f: json.dump(self.portfolio, f, indent=JSON_INDENT)

        # 2. Escribir a DB en shadow mode (OPCIONAL - no bloquea si falla)
        if DB_AVAILABLE:
            # Guardar cash
            db.shadow_write(db.set_state, "cash", self.portfolio["cash"])

            # Guardar posiciones (sync completo)
            for pos_id, pos_data in self.portfolio["positions"].items():
                db.shadow_write(db.upsert_position, pos_id, pos_data)

    def _ensure_log_header(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding='utf-8') as f:
                f.write("Timestamp,Action,Market,Bucket,Price,Shares,Reason,PnL,Cash_After\n")

    def _clean_market_name(self, full_title):
        month_map = {"january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr", "may": "May", "june": "Jun", "july": "Jul", "august": "Aug", "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec"}
        pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d+)'
        matches = re.findall(pattern, full_title, re.IGNORECASE)
        if len(matches) >= DATE_MIN_MATCHES_REQUIRED:
            m1, d1 = matches[0]; m2, d2 = matches[1]
            return f"{month_map.get(m1.lower(), m1[:3])} {d1} - {month_map.get(m2.lower(), m2[:3])} {d2}"
        return "Evento Global"

    def _log_trade(self, action, market, bucket, price, shares, reason, pnl=0.0, strategy="STANDARD",
                   hours_left=None, tweet_count=None, market_consensus=None,
                   pos_id=None, entry_signal_reason=None, exit_signal_reason=None,
                   trade_outcome_label=None, external_event_ref=None, slippage=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_clean = market.replace(",", "")
        reason_clean = reason.replace(",", ".")

        # 1. Escribir a CSV (OBLIGATORIO - fuente de verdad)
        row = f"{ts},{action},{market_clean},{bucket},{price:.3f},{shares:.1f},{reason_clean},{pnl:.2f},{self.portfolio['cash']:.2f}\n"
        with open(self.log_path, "a", encoding='utf-8') as f: f.write(row)

        # 2. Escribir a DB en shadow mode (OPCIONAL - no bloquea si falla)
        if DB_AVAILABLE:
            db.shadow_write(
                db.log_trade,
                action=action,
                market=market,
                bucket=bucket,
                price=price,
                shares=shares,
                reason=reason,
                pnl=pnl,
                cash_after=self.portfolio['cash'],
                mode="PAPER",
                strategy=strategy,
                hours_left=hours_left,
                tweet_count=tweet_count,
                market_consensus=market_consensus,
                pos_id=pos_id,
                entry_signal_reason=entry_signal_reason,
                exit_signal_reason=exit_signal_reason,
                trade_outcome_label=trade_outcome_label,
                external_event_ref=external_event_ref,
                slippage=slippage
            )

    def execute(self, market_title, bucket, signal, price, reason="Manual", strategy_tag="STANDARD",
                hours_left=None, tweet_count=None, market_consensus=None, entry_z_score=None,
                external_event_ref=None, tick_size=None):
        pos_id = f"{market_title}|{bucket}"

        # BUY (Incluye soporte para HEDGE)
        if "BUY" in signal or "FISH" in signal or "HEDGE" in signal:
            if pos_id not in self.portfolio["positions"]:
                # 1. DEFINIR TAMAÑO BASE
                if strategy_tag == "MOONSHOT":
                    base_pct = self.risk_pct_moonshot
                elif "SPREAD" in strategy_tag:
                    base_pct = RISK_PCT_SPREAD
                elif strategy_tag == "CONTRARIAN":
                    base_pct = RISK_PCT_CONTRARIAN
                elif "FISH" in signal:
                    base_pct = self.risk_pct_lotto
                elif "HEDGE" in signal:
                    base_pct = RISK_PCT_HEDGE
                else:
                    base_pct = self.risk_pct_normal

                # 2. 🔥 APLICAR CRITERIO DE KELLY SIMPLIFICADO (SNIPER MODE)
                multiplier = 1.0
                # Solo aumentamos la apuesta si el modelo detecta "Valor" y NO es una cobertura
                if "Val+" in reason and "HEDGE" not in signal:
                    try:
                        edge_val = float(reason.split("Val+")[1].split()[0])
                        if edge_val >= KELLY_EDGE_THRESHOLD_HIGH:
                            multiplier = KELLY_MULTIPLIER_HIGH_EDGE
                        elif edge_val >= KELLY_EDGE_THRESHOLD_MED:
                            multiplier = KELLY_MULTIPLIER_MED_EDGE
                    except: pass

                # 3. CÁLCULO FINAL CON CINTURÓN DE SEGURIDAD
                final_pct = base_pct * multiplier
                final_pct = min(final_pct, MAX_POSITION_SIZE_PCT)
                bet_amount = max(self.portfolio["cash"] * final_pct, self.min_bet)

                if self.portfolio["cash"] >= bet_amount:
                    shares = bet_amount / price
                    self.portfolio["cash"] -= bet_amount
                    self.portfolio["positions"][pos_id] = {
                        "shares": shares, "entry_price": price, "market": market_title,
                        "bucket": bucket, "timestamp": time.time(), "invested": bet_amount,
                        "strategy_tag": strategy_tag, "entry_z_score": entry_z_score
                    }
                    self._save()
                    self._log_trade(signal, market_title, bucket, price, shares, reason,
                                  strategy=strategy_tag, hours_left=hours_left,
                                  tweet_count=tweet_count, market_consensus=market_consensus,
                                  pos_id=pos_id, entry_signal_reason=reason,
                                  external_event_ref=external_event_ref)
                    return f"✅ BUY: ${bet_amount:.2f}"

        # SELL (Incluye TAKE PROFIT y ROTATE)
        elif "SELL" in signal or "DUMP" in signal or "ROTATE" in signal or "TAKE_PROFIT" in signal:
            if pos_id in self.portfolio["positions"]:
                pos = self.portfolio["positions"].pop(pos_id)
                revenue = pos["shares"] * price
                profit = revenue - pos.get("invested", pos["shares"] * pos["entry_price"])
                self.portfolio["cash"] += revenue
                self.portfolio["history"].append({"market": market_title, "profit": profit})

                # Clasificar resultado del trade
                if profit > 0.01:
                    outcome = "profitable"
                elif profit < -0.01:
                    outcome = "loss"
                else:
                    outcome = "break_even"

                # Calcular slippage (diferencia entre precio esperado y ejecutado)
                slippage_val = None
                entry_price = pos.get("entry_price", 0)
                if entry_price > 0:
                    slippage_val = round(price - entry_price, 4)

                # Eliminar posición de DB (con realized_pnl para auditoría)
                if DB_AVAILABLE:
                    db.shadow_write(db.close_position, pos_id, profit)

                self._save()
                self._log_trade(signal, market_title, bucket, price, pos['shares'], reason, pnl=profit,
                              strategy=pos.get("strategy_tag", "STANDARD"), hours_left=hours_left,
                              tweet_count=tweet_count, market_consensus=market_consensus,
                              pos_id=pos_id, exit_signal_reason=reason,
                              trade_outcome_label=outcome, slippage=slippage_val,
                              external_event_ref=external_event_ref)
                return f"💰 SELL: P&L ${profit:.2f}"
        return None

    def print_summary(self, current_prices_data):
        cash = self.portfolio["cash"]; invested = 0.0
        print("\n💼 --- PORTFOLIO ---")
        print(f"   🔹 {'FECHAS EVENTO':<20} | {'BUCKET':<10} | {'PRECIO':<8} | {'ACTUAL':<8} | {'P&L ($)':<8}")
        print("   " + "-"*85)
        for pid, pos in self.portfolio["positions"].items():
            curr_p = pos['entry_price']
            lbl = self._clean_market_name(pos.get('market', ''))
            for m in current_prices_data:
                if self._clean_market_name(m['title']) == lbl:
                    for b in m['buckets']:
                        if str(b['bucket']) == str(pos['bucket']): curr_p = b.get('bid', 0)
            val = pos['shares'] * curr_p
            pnl = val - (pos['shares'] * pos['entry_price'])
            invested += val
            print(f"   🔹 {lbl:<20} | {pos['bucket']:<10} | ${pos['entry_price']:.3f}  | ${curr_p:.3f}  | {pnl:+6.2f}")
        print("   " + "-"*85)
        print(f"   💵 Cash: ${cash:.2f} | 📈 Equity: ${cash+invested:.2f}")

    def get_portfolio(self):
        return self.portfolio
