def run():
    print("\n🤖 ELON-BOT: MARKET MAKER + PAPER TRADING (LOGGING ACTIVADO)")
    print("============================================================")

    brain = HawkesBrain()
    sensor = PolymarketSensor()
    pricer = ClobMarketScanner()
    trader = PaperTrader(initial_cash=1000.0)
    
    last_counts = {}
    last_retrain_time = time.time()
    RETRAIN_INTERVAL = 21600 

    global_events = []
    log_path = os.path.join(DATA_DIR, LIVE_LOG)
    if os.path.exists(log_path):
        try:
            with open(log_path) as f: 
                d = json.load(f)
                global_events = [e for e in d if (time.time()*1000 - e['timestamp']) < 86400000]
        except: pass

    print(f"\n📡 Escuchando... (Mu={brain.params['mu']:.2f}, Alpha={brain.params['alpha']:.2f})")

    while True:
        try:
            # CHEQUEO Y RE-ENTRENAMIENTO AUTOMÁTICO
            if time.time() - last_retrain_time > RETRAIN_INTERVAL:
                brain.load_and_train()
                last_retrain_time = time.time()

            markets = sensor.get_active_counts()
            if not markets: 
                print("💤 No active markets...", end="\r")
                time.sleep(10)
                continue

            tweet_detected = False
            changes = []
            max_diff = 0
            for m in markets:
                curr = m['count']
                prev = last_counts.get(m['id'])
                if prev is None: last_counts[m['id']] = curr; continue
                if curr > prev:
                    diff = curr - prev
                    if diff > max_diff: max_diff = diff
                    changes.append(f"{m['title']}: +{diff}")
                    tweet_detected = True
                last_counts[m['id']] = curr

            now_ms = time.time() * 1000
            if tweet_detected:
                print(f"\n🚨 TWEET DETECTADO! {changes}")
                for _ in range(max_diff): global_events.append({'timestamp': now_ms})
                with open(log_path, 'w') as f: json.dump(global_events, f)
            
            global_events = [e for e in global_events if (now_ms - e['timestamp']) < 86400000]
            
            clob_data = pricer.get_market_prices()
            if clob_data:
                print(f"\n⏱️ {datetime.now().strftime('%H:%M:%S')} | Análisis de Mercado")

            ts_list = [e['timestamp'] for e in global_events]

            # --- FUNCIÓN DE MATCHING INTELIGENTE ---
            def titles_match(tracker_title, market_title):
                # 1. Limpieza básica
                t1 = tracker_title.lower()
                t2 = market_title.lower()
                
                # 2. Si coinciden texto exacto, genial
                if t1 in t2 or t2 in t1: 
                    return True
                
                # 3. LA CLAVE: Comparar solo los NÚMEROS (las fechas)
                # Extraemos '9', '16', '2026', etc.
                nums1 = set(re.findall(r'\d+', t1))
                nums2 = set(re.findall(r'\d+', t2))
                
                # Si comparten al menos 2 números (ej: día inicio y día fin), es match
                common = nums1.intersection(nums2)
                if len(common) >= 2:
                    return True
                return False
            # ---------------------------------------
            
            for m_poly in markets:
                relevant_prices = next((p for p in clob_data if titles_match(m_poly['title'], p['title'])), None)
                if not relevant_prices: continue

                # ==============================================================================
                # 🛑 FIX UNIVERSAL: AJUSTE DE TIEMPO Y PRUDENCIA
                # ==============================================================================
                # Asumimos que TODOS los mercados de tweets cierran a las 12:00 PM ET.
                # Esto significa que el último día solo tiene 12 horas útiles, no 24.
                # Además, aplicamos un margen de seguridad del 15% para evitar "Long Shots" imposibles.
                
                hours_to_predict = m_poly['hours']

                # 1. CORTE DE MEDIODÍA (GLOBAL):
                # Restamos 12 horas a todos los eventos.
                # Si el evento cierra a mediodía, esto ajusta la realidad.
                # Si cerrase a medianoche, esto nos hace ser conservadores (que es bueno).
                hours_to_predict = hours_to_predict - 12.0

                # 2. FACTOR DE REALIDAD (Recorte del 15%):
                # Obligamos al bot a "ganar la apuesta" en menos tiempo del real.
                # Esto mata las probabilidades de buckets extremos (740+, 580+) si no va sobrado.
                hours_to_predict = hours_to_predict * 0.85

                # Protección para evitar cuelgues con números negativos
                if hours_to_predict < 0.1: hours_to_predict = 0.1

                # Ejecutamos la predicción
                sims = brain.predict(ts_list, hours_to_predict)
                # ==============================================================================

                final_sims = np.array([m_poly['count'] + s for s in sims])
                total_s = len(final_sims)

                print("-" * 75)
                print(f"\n>>> {m_poly['title']} [Actual: {m_poly['count']}] (Real: {m_poly['hours']:.1f}h | Bot: {hours_to_predict:.1f}h)")
                print("-" * 75)

                print("    Distribución por Buckets (Mercado):")
                bucket_stats = []
                max_prob = 0
                for b in relevant_prices['buckets']:
                    count = sum(1 for x in final_sims if b['min'] <= x <= b['max'])
                    prob = count / len(final_sims)
                    if prob > max_prob: max_prob = prob
                    bucket_stats.append({'label': b['bucket'], 'prob': prob})

                for item in bucket_stats:
                    if item['prob'] > 0.005: 
                        bar_len = int((item['prob'] / max_prob) * 30) if max_prob > 0 else 0
                        bar = "█" * bar_len
                        icon = "⭐" if item['prob'] == max_prob else ""
                        print(f"    {item['label']:<10} | {bar} ({item['prob']*100:.1f}%) {icon}")
                
                print("-" * 75)
                print(f"{'BUCKET':<10} | {'BID':<8} | {'ASK':<8} | {'FAIR':<8} | {'ACCIÓN':<10} | {'MOTIVO'}")

                for b in relevant_prices['buckets']:
                    matches = sum(1 for x in final_sims if b['min'] <= x <= b['max'])
                    fair_val = matches / total_s
                    ask = b.get('ask', 0)
                    bid = b.get('bid', 0)
                    
                    action = "-"
                    reason = "_"

                    # ==============================================================================
                    # 🚨 ALARMA DE RUPTURA INTELIGENTE (TIEMPO vs ESPACIO)
                    # ==============================================================================
                    # Calculamos el espacio que nos queda antes de que el bucket "explote"
                    tweets_left_space = b['max'] - m_poly['count']
                    
                    # Verificamos si estamos DENTRO del bucket ganador ahora mismo
                    is_in_the_money = (m_poly['count'] >= b['min'] and m_poly['count'] <= b['max'])
                    
                    if is_in_the_money:
                        # FACTOR DE RIESGO: ¿Cuántos tweets por hora se necesitan para romperlo?
                        # Usamos las HORAS REALES (m_poly['hours']), no las simuladas.
                        # Si quedan 17h, necesitamos al menos 17 * 2.5 = 42 tweets de espacio para estar tranquilos.
                        
                        safe_buffer_needed = m_poly['hours'] * 2.5  # Asumimos ~2.5 tweets/hora de media
                        
                        # Excepción: Si queda menos de 1 hora, relajamos el buffer (final de partido)
                        if m_poly['hours'] < 1.0: safe_buffer_needed = 2 

                        # SI el espacio es menor que el necesario... VENDEMOS CORRIENDO.
                        if tweets_left_space < safe_buffer_needed and bid > 0.005:
                            action = "🔴 SELL"
                            # Ejemplo Log: "Riesgo Ruptura (Quedan 11 slots para 17.5h)"
                            reason = f"Riesgo Ruptura ({tweets_left_space} slots / {m_poly['hours']:.1f}h)"
                            fair_val = 0.00 # Forzamos a que el bot vea que esto vale 0

                    # ==============================================================================

                    # --- Lógica Estándar (solo si no hay emergencia) ---
                    elif ask > 0 and fair_val > (ask + 0.10): 
                        action = f"🟢 BUY"
                        diff = fair_val - ask
                        reason = f"Edge +{diff:.2f} (Barato)"
                    elif bid > 0 and fair_val < (bid - 0.05): 
                        action = f"🔴 SELL"
                        diff = bid - fair_val
                        reason = f"Sobreprecio +{diff:.2f}"
                    
                    if m_poly['hours'] < 4 and fair_val < 0.01 and bid > 0.05:
                         action = "💀 DUMP"
                         reason = "Zombie (Time Decay)"
                    
                    if ask > 0.01 or fair_val > 0.01:
                        print(f"{b['bucket']:<10} | {bid:.3f}    | {ask:.3f}    | {fair_val:.3f}    | {action:<10} | {reason}")

                    if action != "-" and action != "💀 DUMP": 
                        clean_action = action.split()[1] if " " in action else action
                        # PASAMOS EL REASON AL TRADER PARA EL LOG
                        trade_res = trader.execute(m_poly['title'], b['bucket'], action, ask if "BUY" in action else bid, reason=reason)
                        if trade_res:
                            print(f"   👉 {trade_res}")

            if clob_data:
                trader.print_summary(clob_data)
                
            if not clob_data: print(".", end="", flush=True)
            time.sleep(0.5)

        except KeyboardInterrupt: break
        except Exception as e: 
            print(f"Error Loop: {e}")
            time.sleep(1)