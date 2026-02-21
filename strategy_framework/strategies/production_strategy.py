"""
Production Strategy - EXACT COPY of elon_auto_bot_threads.py logic

This strategy replicates the EXACT prediction and trading logic from the production bot
to ensure backtest results match real-world performance.

Based on: elon_auto_bot_threads.py lines 567-870
"""

import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
import re
from scipy.stats import norm
from .base_strategy import BaseStrategy, Signal, SignalType, MarketState, Position
from ..utils import detect_event_type
from ..production_config import (
    MAX_Z_SCORE_ENTRY, MIN_PRICE_ENTRY, MIN_EDGE,
    ENABLE_CLUSTERING, CLUSTER_RANGE,
    CLUSTER_MULTIPLIER_SHORT_EARLY, CLUSTER_MULTIPLIER_SHORT_LATE,
    MARKET_CONSENSUS_WEIGHT,
    STOP_LOSS_NORMAL, STOP_LOSS_CHEAP_ENTRY, STOP_LOSS_CHEAP_THRESHOLD,
    STOP_LOSS_LATE_GAME, STOP_LOSS_Z_MIN, VICTORY_LAP_TIME_HOURS,
    get_safety_margin, is_impossible_to_reach, is_warmup
)


class ProductionStrategy(BaseStrategy):
    """
    EXACT replica of production bot logic (V12.16 + V13 Sigma Decay + Moonshot V32)

    Extracted from elon_auto_bot_threads.py to ensure 100% consistency
    between backtest and live trading.
    """

    def __init__(self, config: Dict = None):
        # Use production config, allow override for testing
        self.max_z_score_entry = MAX_Z_SCORE_ENTRY
        self.min_price_entry = MIN_PRICE_ENTRY
        self.min_edge = MIN_EDGE
        self.enable_clustering = ENABLE_CLUSTERING
        self.cluster_range = CLUSTER_RANGE

        # Hawkes tracking
        self.hawkes_activations = 0
        self.hawkes_wins = 0
        
        # Moonshot tracking
        self.moonshot_executions = 0
        self.moonshot_cooldowns = {}  # 🛰️ Cooldowns para evitar recompras inmediatas

        # Override with config if provided
        if config:
            self.max_z_score_entry = config.get("max_z_score_entry", MAX_Z_SCORE_ENTRY)
            self.min_price_entry = config.get("min_price_entry", MIN_PRICE_ENTRY)
            self.min_edge = config.get("min_edge", MIN_EDGE)
            self.enable_clustering = config.get("enable_clustering", ENABLE_CLUSTERING)
            self.cluster_range = config.get("cluster_range", CLUSTER_RANGE)

        super().__init__(config or {})
        # Override name after super().__init__
        self.name = "ProductionStrategy"

    def _calculate_prediction(self, market_state: MarketState) -> tuple[float, float]:
        """
        EXACT COPY of production prediction logic (V22 UNIVERSAL SCALE)

        Auto-learns bucket sizes for any market (Elon, Bitcoin, etc)
        """
        try:
            # --- 1. BASE DATA ---
            p_count = market_state.count
            p_hours_left = market_state.hours_left
            p_avg_hist = market_state.daily_avg

            # Event Duration Fix
            if p_count > 130 or p_hours_left > 72:
                total_duration = 168.0
            elif p_hours_left < 40:
                total_duration = 48.0
            else:
                total_duration = 72.0
                
            hours_elapsed = max(1.0, total_duration - p_hours_left)
            rate_actual_diario = (p_count / hours_elapsed) * 24.0
            
            # --- 2. FATIGUE CALCULATION (GRAVITY) ---
            if p_hours_left < 6.0:
                fatigue_weight = 0.95
            elif p_hours_left < 24.0:
                fatigue_weight = 0.75
            else:
                fatigue_weight = 0.35

            # --- 3. PURE PROJECTION ---
            projected_rate = (rate_actual_diario * fatigue_weight) + (p_avg_hist * (1 - fatigue_weight))
            
            # Safety cap for long term
            if p_hours_left > 24.0:
                max_rate_allowed = p_avg_hist * 3.0
                projected_rate = min(projected_rate, max_rate_allowed)

            mean_prediction = p_count + (projected_rate / 24.0 * p_hours_left)
            
            # --- 4. REALITY CHECK (AUTO-SCALING) ---
            final_mean = mean_prediction
            
            try:
                # Get buckets with real liquidity
                all_buckets = market_state.buckets
                market_buckets = [b for b in all_buckets if b.get('bid', 0) > 0.05]
                
                if market_buckets and p_hours_left > 12.0:
                    # A) LEARNING: What's the step size in this market?
                    # (Ex: Elon = 20, Bitcoin = 1000)
                    bucket_sizes = []
                    for b in all_buckets:
                        size = b['max'] - b['min']
                        # Filter out infinites (which usually have giant max values)
                        if size < 100000 and size > 0:
                            bucket_sizes.append(size)
                    
                    # Calculate median (the standard step size)
                    if bucket_sizes:
                        bucket_sizes.sort()
                        avg_step = bucket_sizes[len(bucket_sizes)//2]
                    else:
                        avg_step = 10.0  # Safety fallback
                    
                    # B) CONSENSUS CALCULATION
                    w_sum = 0
                    w_vol = 0
                    
                    for b in market_buckets:
                        # Detect if it's an infinite bucket by comparing to standard step
                        range_size = b['max'] - b['min']
                        
                        if range_size > (avg_step * 5.0):
                            # It's an infinite bucket (ex: 580+)
                            # Use learned step to estimate center
                            mid_val = b['min'] + avg_step
                        else:
                            # It's a normal bucket
                            mid_val = (b['min'] + b['max']) / 2
                        
                        w_sum += mid_val * b['bid']
                        w_vol += b['bid']

                    if w_vol > 0:
                        consensus = w_sum / w_vol
                        # If we deviate >15%, correct towards market
                        if abs(final_mean - consensus) > (consensus * 0.15):
                            final_mean = (final_mean * 0.6) + (consensus * 0.4)
                            
            except Exception as e_mkt:
                # If market fails, continue with our prediction
                pass

        except Exception as e:
            final_mean = p_count + (p_avg_hist / 24.0 * p_hours_left)

        # --- 5. ADAPTIVE SIGMA ---
        raw_sigma = (final_mean ** 0.5) * 1.5
        time_factor = (market_state.hours_left / 168.0) ** 0.5
        eff_std = max(raw_sigma * max(0.3, time_factor), 3.0)

        return final_mean, eff_std

        # --- SYNTHETIC SIGMA (ADAPTIVE) ---
        # 1. Base volatility (Square root of mean * Brain factor)
        raw_sigma = (final_mean ** 0.5) * eff_std_factor if final_mean > 0 else 5.0
        
        # 2. Sigma Decay (Uncertainty drops as time runs out)
        time_factor = (market_state.hours_left / 168.0) ** 0.5
        time_factor = max(0.2, min(1.0, time_factor))
        
        # 3. Final calculation with floor
        eff_std = max(raw_sigma * time_factor, 3.0)

        return final_mean, eff_std

    def _apply_sigma_decay(self, eff_std: float, hours_left: float) -> float:
        """
        V13 Sigma Decay (lines 695-707)

        Collapse variance as time runs out
        """
        # Decay factor (square root of time ratio)
        decay_factor = (hours_left / 72.0) ** 0.5

        # Clamp to minimum 25% (safety floor)
        decay_factor = max(0.25, min(1.0, decay_factor))

        # Apply decay
        decayed_std = eff_std * decay_factor

        return decayed_std

    def analyze(self, market_state: MarketState, current_positions: List[Position]) -> List[Signal]:
        """
        Generate signals using EXACT production logic
        """
        signals = []

        # Get prediction
        pred_mean, eff_std = self._calculate_prediction(market_state)

        # Get owned buckets for clustering
        my_buckets = self._get_owned_bucket_values(current_positions, market_state.title)
        
        # Get owned bucket IDs for SMART PROXIMITY logic
        my_buckets_ids = self._get_owned_bucket_ids(current_positions, market_state.title)

        for bucket in market_state.buckets:
            # Skip already passed buckets
            if bucket['max'] < market_state.count:
                continue

            # Get prices
            bid = bucket.get('bid', 0)
            ask = bucket.get('ask', 0)

            # Calculate bucket mid
            if bucket['max'] >= 99999:
                mid = bucket['min'] + 20
            else:
                mid = (bucket['min'] + bucket['max']) / 2

            # Apply V13 Sigma Decay (lines 693-707)
            decayed_std = self._apply_sigma_decay(eff_std, market_state.hours_left)

            # Calculate z-score with decayed sigma (line 712)
            z_score = abs(mid - pred_mean) / decayed_std if decayed_std > 0 else 999

            # Calculate fair value with decayed sigma (lines 713-716)
            p_min = norm.cdf(bucket['min'], pred_mean, decayed_std)
            if bucket['max'] >= 99999:
                fair = 1.0 - p_min
            else:
                fair = norm.cdf(bucket['max'] + 1, pred_mean, decayed_std) - p_min

            # Check if we own this bucket
            owned_pos = self._find_position(current_positions, bucket['bucket'])

            if owned_pos:
                # Check exit conditions (pass bucket IDs for SMART PROXIMITY)
                exit_signal = self._check_exit_conditions(
                    bucket, owned_pos, market_state, pred_mean, decayed_std, z_score, fair, my_buckets_ids
                )
                if exit_signal:
                    signals.append(exit_signal)
            else:
                # Check entry conditions (lines 824-869)
                entry_signal = self._check_entry_conditions(
                    bucket, market_state, my_buckets, my_buckets_ids, pred_mean, decayed_std, z_score, fair
                )
                if entry_signal:
                    signals.append(entry_signal)

        # 🛰️ MOONSHOT SATELITAL (Al final del ciclo)
        moonshot_signal = self._check_moonshot_opportunity(
            market_state, current_positions, pred_mean
        )
        if moonshot_signal:
            signals.append(moonshot_signal)

        return signals

    def _check_entry_conditions(
        self,
        bucket: Dict,
        market_state: MarketState,
        my_buckets: List[float],
        my_buckets_ids: List[str],
        pred_mean: float,
        decayed_std: float,
        z_score: float,
        fair: float
    ) -> Signal:
        """
        EXACT entry logic from production (lines 824-869)
        + FIX 2: Dynamic clustering for short events
        """
        ask = bucket.get('ask', 0)
        bucket_headroom = bucket['max'] - market_state.count
        hours_left = market_state.hours_left

        # Warmup check (line 681, 833)
        if is_warmup(market_state.count, hours_left):
            return None

        # Anti-kamikaze filter (lines 828-844)
        buy_safety = get_safety_margin(hours_left)

        if bucket_headroom < buy_safety:
            return None

        # Reality check (lines 847-851)
        if is_impossible_to_reach(market_state.count, bucket['min'], hours_left):
            return None

        # Entry threshold: z <= 0.85 (line 791)
        # 🟢 ESTRATEGIA DE ACUMULACIÓN (+41% ROI RESTORED)
        # Sin bloqueo de vecinos - permite construir clusters naturales
        if z_score <= self.max_z_score_entry and ask >= self.min_price_entry:
            edge = fair - ask
            # CAMBIO #10: Dynamic Min Edge
            dynamic_min_edge = self.min_edge + (decayed_std * 0.01)
            if edge > dynamic_min_edge:
                # FIX 2: Dynamic clustering for short events
                passes_clustering = True
                if self.enable_clustering and my_buckets_ids:
                    # Detect event type and calculate dynamic cluster range
                    event_type, bucket_size_detected = detect_event_type(
                        market_state.metadata, 
                        market_state.buckets
                    )

                    if event_type == 'short':
                        # SHORT EVENTS: Use dynamic clustering based on time
                        if hours_left > 24:
                            multiplier = CLUSTER_MULTIPLIER_SHORT_EARLY  # 1.5x buckets
                        else:
                            multiplier = CLUSTER_MULTIPLIER_SHORT_LATE   # 1.0x buckets

                        # Use detected bucket size or default
                        avg_bucket_size = bucket_size_detected if bucket_size_detected else 24
                        cluster_range = avg_bucket_size * multiplier
                    else:
                        # LONG EVENTS: Use fixed cluster range (already works well)
                        cluster_range = self.cluster_range

                    # Check distance to existing positions
                    try:
                        if "+" in bucket['bucket']:
                            new_min = int(bucket['bucket'].replace("+", ""))
                        else:
                            new_min = int(bucket['bucket'].split("-")[0])

                        for owned_bucket in my_buckets_ids:
                            try:
                                if "+" in owned_bucket:
                                    owned_min = int(owned_bucket.replace("+", ""))
                                else:
                                    owned_min = int(owned_bucket.split("-")[0])

                                distance = abs(new_min - owned_min)
                                if distance > cluster_range:
                                    passes_clustering = False
                                    break
                            except:
                                pass
                    except:
                        pass

                if passes_clustering:
                    return Signal(
                        type=SignalType.BUY,
                        market_title=market_state.title,
                        bucket=bucket['bucket'],
                        price=ask,
                        confidence=min(edge * 2, 1.0),
                        reason=f"Val+{edge:.2f}",
                        metadata={"z_score": z_score, "fair": fair},
                        strategy_tag="STANDARD"  # Normal accumulation trade
                    )

        return None

    def _check_exit_conditions(
        self,
        bucket: Dict,
        position: Position,
        market_state: MarketState,
        pred_mean: float,
        decayed_std: float,
        z_score: float,
        fair: float,
        my_buckets_ids: List[str] = None
    ) -> Signal:
        """
        Exit logic V12.24 - STATIC SAFETY THRESHOLDS + MOONSHOT IMMUNITY + SMART PROXIMITY
        """
        if my_buckets_ids is None:
            my_buckets_ids = []
        bid = bucket.get('bid', 0)
        entry = position.entry_price
        profit_pct = (bid - entry) / entry if entry > 0 else 0

        # ==========================================================
        # 🛡️ INMUNIDAD POR ETIQUETA (DNA TAGGING)
        # ==========================================================
        # Si es Moonshot, aplicamos lógica de lotería + trailing stop
        if position.strategy_tag == 'MOONSHOT':
            # 💀 EXPIRACIÓN: Si bid va a $0, realizar pérdida
            if bid <= 0.01:
                # 🛰️ Activar cooldown de 24h después de pérdida total
                self.moonshot_cooldowns[bucket['bucket']] = datetime.now() + timedelta(hours=24)

                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=1.0,
                    reason=f"Moonshot Expired (Total Loss)",
                    metadata={"z_score": z_score, "profit_pct": profit_pct, "entry": entry}
                )

            # 🏆 VICTORIA LAP: Si llega a $0.99, vendemos
            if bid >= 0.99:
                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=0.98,
                    reason=f"Moonshot Victory Lap (${bid:.2f})",
                    metadata={"z_score": z_score, "profit_pct": profit_pct}
                )

            # CAMBIO #4: Moonshot Exit Parcial Temprano
            if 0.20 <= bid <= 0.30 and profit_pct >= 3.0:
                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=0.80,
                    reason=f"Moonshot Partial Exit (Lock {profit_pct*100:.0f}%, ${bid:.2f})",
                    metadata={"z_score": z_score, "profit_pct": profit_pct, "entry": entry}
                )

            # UPDATE: Trackear precio máximo visto
            current_max = position.max_price_seen or entry
            if bid > current_max:
                current_max = bid
                position.max_price_seen = bid

            # 🎯 TRAILING STOP ADAPTATIVO PARA MOONSHOTS
            if current_max >= 0.50:
                # Una vez que llegamos a $0.50+, activamos trailing stop
                # Si baja $0.15 desde el peak, cerramos con ganancias
                drawdown_from_peak = current_max - bid

                if drawdown_from_peak >= 0.15:
                    # 🛰️ Cooldown de 48h después de trailing stop (capturamos ganancia)
                    self.moonshot_cooldowns[bucket['bucket']] = datetime.now() + timedelta(hours=48)

                    return Signal(
                        type=SignalType.SELL,
                        market_title=market_state.title,
                        bucket=bucket['bucket'],
                        price=bid,
                        confidence=0.95,
                        reason=f"Moonshot Trailing Stop (Peak ${current_max:.2f} → ${bid:.2f}, -{drawdown_from_peak:.2f})",
                        metadata={"z_score": z_score, "profit_pct": profit_pct, "peak": current_max}
                    )
                else:
                    # Dejamos correr ganancias
                    return None
            else:
                # 🛡️ INMUNIDAD TOTAL hasta $0.50
                # No permitimos que ninguna otra regla venda el moonshot
                return None  # HOLD

        bucket_headroom = bucket['max'] - market_state.count
        hours_left = max(0.1, market_state.hours_left)

        # CAMBIO #5: Proximity Danger Dinámico
        if hours_left > 24.0:
            base_threshold = 15
        elif hours_left > 12.0:
            base_threshold = 12
        elif hours_left > 6.0:
            base_threshold = 10
        else:
            base_threshold = 8

        volatility_buffer = int(decayed_std * 1.5)
        safety_threshold = base_threshold + volatility_buffer

        # --- SMART PROXIMITY: Check if we have coverage above ---
        if bucket_headroom < safety_threshold and bucket_headroom >= 0:
            # 1. Calculate if we're covered above (have the next neighbor bucket)
            is_covered_above = False
            next_neighbor_min = bucket['max'] + 1
            
            for owned_b in my_buckets_ids:
                try:
                    if "+" in owned_b:
                        o_min = int(owned_b.replace("+", ""))
                    else:
                        o_min = int(owned_b.split("-")[0])
                    
                    if o_min == next_neighbor_min:
                        is_covered_above = True
                        break
                except:
                    pass
            
            # 2. Only sell if NOT covered (exposed upside)
            if not is_covered_above:
                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=1.0,
                    reason=f"Proximity Danger ({bucket_headroom} left)",
                    metadata={"z_score": z_score, "profit_pct": profit_pct}
                )
            # If covered, we hold (no signal returned)
        
        # B) VICTORY LAP (line 737)
        if hours_left <= 48.0 and bid > 0.95:
            return Signal(
                type=SignalType.SELL,
                market_title=market_state.title,
                bucket=bucket['bucket'],
                price=bid,
                confidence=0.9,
                reason=f"Victory Lap (Price {bid:.2f} > 0.95)",
                metadata={"z_score": z_score, "profit_pct": profit_pct}
            )
        
        # ------------------------------------------------------------------------------
        # REGLAS DE TRADING ESTÁNDAR (Profit Taking, Stop Loss...)
        # ------------------------------------------------------------------------------
        if hours_left > 24.0:
            profit_threshold = 1.8 if hours_left > 48.0 else 2.4

            # Tesoro Paranoico
            if profit_pct > 1.5 and z_score > 0.9:
                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=0.85,
                    reason="Paranoid Treasure (Secured)",
                    metadata={"z_score": z_score, "profit_pct": profit_pct}
                )

            # Protección de Beneficios
            if profit_pct > 0.05 and z_score > profit_threshold:
                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=0.8,
                    reason=f"Protect Profit (Mid-Game Z{profit_threshold})",
                    metadata={"z_score": z_score, "profit_pct": profit_pct}
                )

            # Pánico Global (V16 - ANTI-CHURNING EXTREMO)
            # Solo vendemos si el mercado está ROTO (Z > 8.0).
            # Umbral extremo para evitar vender en volatilidad normal (Z 2-5 en backtest).
            if z_score > 8.0 and profit_pct < 0.10:
                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=0.9,
                    reason="Extreme Panic (Z>8)",
                    metadata={"z_score": z_score, "profit_pct": profit_pct}
                )

            # Stop Loss Adaptativo
            avg_entry = bid / (1 + profit_pct) if (1 + profit_pct) != 0 else bid
            
            # Si entramos barato (< $0.12), NUNCA vendemos por pérdidas (Spread Filter)
            if avg_entry < STOP_LOSS_CHEAP_THRESHOLD: 
                sl_limit = STOP_LOSS_CHEAP_ENTRY  # -200% "Manos de Diamante"
            else: 
                sl_limit = STOP_LOSS_NORMAL  # -40% para trades normales
            
            if hours_left < VICTORY_LAP_TIME_HOURS:
                sl_limit = STOP_LOSS_LATE_GAME  # Disable stop loss in late game

            if profit_pct < sl_limit and z_score > STOP_LOSS_Z_MIN:
                # Stop Loss does NOT trigger cooldown anymore
                return Signal(
                    type=SignalType.SELL,
                    market_title=market_state.title,
                    bucket=bucket['bucket'],
                    price=bid,
                    confidence=0.7,
                    reason=f"Stop Loss Adaptativo (Hit {profit_pct*100:.1f}%)",
                    metadata={"z_score": z_score, "profit_pct": profit_pct}
                )

        return None

    def calculate_position_size(
        self,
        signal: Signal,
        available_cash: float,
        portfolio_value: float
    ) -> float:
        """
        🔥 Position sizing with Kelly Criterion (Sniper Mode)
        
        Matches EXACT logic from elon_auto_bot_threads.py
        - Base sizing by strategy type
        - Kelly multiplier based on edge value
        - Hard cap at 10% max risk
        - HEDGE support for defensive positions
        """
        from ..production_config import RISK_PCT_NORMAL, RISK_PCT_MOONSHOT, MIN_BET

        # 1. DEFINIR TAMAÑO BASE
        if signal.strategy_tag == "MOONSHOT":
            base_pct = RISK_PCT_MOONSHOT  # 1%
        elif "FISH" in signal.type.value:
            base_pct = 0.01  # 1% for lotto/fish plays
        elif "HEDGE" in signal.type.value:  # <--- NUEVO: TAMAÑO PARA COBERTURA
            base_pct = 0.025  # 2.5% del capital (Seguro barato)
        else:
            base_pct = RISK_PCT_NORMAL  # 4% (Standard)

        # 2. 🔥 APLICAR CRITERIO DE KELLY SIMPLIFICADO (SNIPER MODE)
        multiplier = 1.0
        
        # Solo aumentamos la apuesta si el modelo detecta "Valor" y NO es una cobertura
        if "Val+" in signal.reason and "HEDGE" not in signal.type.value:
            try:
                # Extraemos el número del texto "Val+0.36" -> 0.36
                edge_val = float(signal.reason.split("Val+")[1].split()[0])
                
                # ESCALERA DE CONVICCIÓN
                if edge_val >= 0.40:    # Si el mercado nos regala 40 centavos o más
                    multiplier = 2.0    # Doble apuesta (8%)
                elif edge_val >= 0.20:  # Si nos regala 20 centavos
                    multiplier = 1.5    # +50% apuesta (6%)
                    
            except:
                pass
        
        # 3. CÁLCULO FINAL CON CINTURÓN DE SEGURIDAD
        final_pct = base_pct * multiplier
        
        # 🛡️ HARD CAP: Nunca arriesgar más del 10% del portfolio en una sola bala
        # Esto te protege de errores de código o cisnes negros.
        final_pct = min(final_pct, 0.10)
        
        bet_amount = max(available_cash * final_pct, MIN_BET)

        # Never exceed available cash
        if bet_amount > available_cash:
            bet_amount = available_cash

        return bet_amount

    def _find_position(self, positions: List[Position], bucket: str) -> Position:
        """Find position for specific bucket"""
        for pos in positions:
            if pos.bucket == bucket:
                return pos
        return None

    def _get_owned_bucket_values(self, positions: List[Position], market_title: str) -> List[float]:
        """Get midpoint values of owned buckets for clustering"""
        vals = []
        for pos in positions:
            if market_title.lower() in pos.market.lower():
                try:
                    if "+" in pos.bucket:
                        mid = int(pos.bucket.replace('+', '')) + 20
                    else:
                        nums = [int(n) for n in pos.bucket.split('-')]
                        mid = sum(nums) / 2
                    vals.append(mid)
                except:
                    pass
        return vals
    
    def _get_owned_bucket_ids(self, positions: List[Position], market_title: str) -> List[str]:
        """Get bucket IDs (strings) of owned buckets for SMART PROXIMITY logic"""
        bucket_ids = []
        for pos in positions:
            if market_title.lower() in pos.market.lower():
                bucket_ids.append(pos.bucket)
        return bucket_ids

    def _check_moonshot_opportunity(
        self,
        market_state: MarketState,
        current_positions: List[Position],
        pred_mean: float
    ) -> Signal:
        """
        🛰️ MÓDULO MOONSHOT (SATÉLITE) - V33 (CON ETIQUETADO DNA)
        
        Estrategia secundaria que opera SOLO al final del ciclo.
        Busca 'Cisnes Negros' al alza (buckets lejanos y baratos).
        Presupuesto: Max 2 buckets.
        Rango de precio: $0.05 - $0.09 (Donde nadie apuesta).
        
        EXACT COPY from elon_auto_bot_threads.py lines 64-159
        """
        try:
            # 1. Filtro de Seguridad Dinámico: Timing basado en duración del evento
            p_hours_left = market_state.hours_left
            p_count = market_state.count
            
            # Determinamos duración total del evento
            if p_count > 130 or p_hours_left > 72:
                total_duration = 168.0  # 7 días
            elif p_hours_left < 40:
                total_duration = 48.0   # 2 días
            else:
                total_duration = 72.0   # 3 días
            
            # 1b. RESTRICCIÓN: Moonshots solo en eventos largos (≥3 días)
            # En eventos cortos hay menos buckets y menos tiempo para materializar
            if total_duration < 72.0:
                return None  # Evento demasiado corto para moonshots
            
            # Moonshots solo en el primer 40% del evento (cuando queda 60%+)
            min_hours_required = total_duration * 0.60
            
            if p_hours_left < min_hours_required or p_count < 35:
                return None  # Demasiado tarde o muy temprano

            # 2. Revisar Cartera ACTUAL (Buscamos Moonshots por ETIQUETA, no adivinanza)
            moonshots_count = 0
            moonshot_buckets_ids = []
            
            base_proj = market_state.count + (market_state.daily_avg / 24.0 * market_state.hours_left)
            
            for pos in current_positions:
                # Verificamos si es de este evento
                clean_market = ''.join(filter(str.isalnum, pos.market.lower()))
                clean_poly = ''.join(filter(str.isalnum, market_state.title.lower()))
                
                if clean_poly in clean_market or clean_market in clean_poly:
                    # AQUÍ ESTÁ LA CLAVE: Solo contamos si es ESTRATEGIA MOONSHOT
                    if pos.strategy_tag == 'MOONSHOT':
                        moonshots_count += 1
                        moonshot_buckets_ids.append(pos.bucket)

            # 3. Límite de Inventario (Máximo 2 Moonshots)
            if moonshots_count >= 2: 
                return None  # Cupo lleno

            # 4. Definir Objetivo (Rage Mode: +120 tweets sobre la media)
            rage_target = base_proj + 120.0

            # 4b. CAMBIO #4: Límite de Realismo más estricto
            max_reasonable_distance = market_state.daily_avg * 2.0  # Max 2x la media diaria

            # 5. Buscar Candidatos
            candidates = []

            for b in market_state.buckets:
                ask = b.get('ask', 0)

                # --- FILTROS ESTRICTOS ---
                # A) CAMBIO #4: Precio ampliado para más oportunidades
                if not (0.005 <= ask <= 0.011):
                    continue
                
                # B) Dirección: Solo buckets SUPERIORES a la proyección (Apuesta Alza)
                if b['min'] < base_proj: 
                    continue
                
                # B2) Realismo: No apostar demasiado lejos (evita moonshots imposibles)
                if b['min'] - base_proj > max_reasonable_distance:
                    continue
                
                # C) Que NO lo tengamos ya
                if b['bucket'] in moonshot_buckets_ids: 
                    continue
                
                # C2) 🛰️ COOLDOWN CHECK: No recomprar moonshots vendidos recientemente
                if b['bucket'] in self.moonshot_cooldowns:
                    if datetime.now() < self.moonshot_cooldowns[b['bucket']]:
                        continue  # Aún en cooldown
                    else:
                        # Cooldown expirado, limpiamos
                        del self.moonshot_cooldowns[b['bucket']]
                
                # D) Distancia al Rage Target
                if b['max'] >= 99999: 
                    mid = b['min'] + 20
                else: 
                    mid = (b['min'] + b['max']) / 2
                
                dist = abs(mid - rage_target)
                
                # Si está razonablemente cerca del objetivo (+/- 40 tweets)
                if dist < 40:
                    candidates.append({'bucket': b, 'dist': dist, 'ask': ask})
            
            # 6. EJECUCIÓN (Solo el MEJOR candidato)
            if candidates:
                # Ordenamos por cercanía al target
                candidates.sort(key=lambda x: x['dist'])
                best = candidates[0]
                
                self.moonshot_executions += 1
                
                return Signal(
                    type=SignalType.BUY,
                    market_title=market_state.title,
                    bucket=best['bucket']['bucket'],
                    price=best['ask'],
                    confidence=0.5,  # Lower confidence for moonshots
                    reason="Moonshot V33",
                    metadata={
                        "dist_to_target": best['dist'],
                        "rage_target": rage_target,
                        "base_proj": base_proj
                    },
                    strategy_tag="MOONSHOT"  # DNA tag for immunity
                )

        except Exception as e:
            # Silent fail like production
            pass
            
        return None
