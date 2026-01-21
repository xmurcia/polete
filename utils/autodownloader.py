import os
import requests
import pandas as pd
from datetime import datetime, timezone

# ==========================================
# CLASE: XTrackerDownloader (Auto-Sync)
# ==========================================
class XTrackerDownloader:
    def __init__(self, data_dir):
        self.base_url = "https://xtracker.polymarket.com/api"
        self.user_id = "c4e2a911-36ec-4453-8a39-1edb5e6b2969"  # Elon Musk ID
        self.data_dir = data_dir
        self.user = 'elonmusk'
        
        # Crear directorio si no existe
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.session = requests.Session()
        # Headers "humanos" para evitar bloqueos
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        })

    def sync_history(self):
        print(f"📥 XTracker Sync: Buscando historial cerrado...", end=" ")
        try:
            # 1. Obtener lista de eventos
            resp = self.session.get(f"{self.base_url}/users/{self.user}", timeout=10)
            if resp.status_code != 200:
                print(f"❌ Error API: {resp.status_code}")
                return

            trackings = resp.json().get('data', {}).get('trackings', [])
            # Ordenamos: Los que terminan más tarde primero
            trackings.sort(key=lambda x: x.get('endDate', ''), reverse=True)
            
            # 2. BUSCAR EL PRIMER EVENTO YA TERMINADO
            target_event = None
            now_utc = datetime.now(timezone.utc)

            for t in trackings:
                try:
                    # Parseamos la fecha de fin (ISO 8601)
                    end_date_str = t['endDate'].replace('Z', '+00:00')
                    end_dt = datetime.fromisoformat(end_date_str)
                    
                    # CONDICIÓN CLAVE: Solo si el evento ya acabó (es pasado)
                    if end_dt < now_utc:
                        target_event = t
                        break # Encontramos el último evento cerrado
                except:
                    continue
            
            if not target_event:
                print("⚠️ No se encontraron eventos finalizados recientes.")
                return

            title = target_event['title']
            print(f"\n   🎯 Objetivo Histórico: {title}")
            print(f"      (Periodo: {target_event['startDate']} -> {target_event['endDate']})")

            # ==================================================================
            # 3. EL TRUCO PARA QUE FUNCIONE COMO CURL (URL MANUAL)
            # ==================================================================
            # Requests codifica los ':' como '%3A'. XTracker a veces odia eso.
            # Construimos la Query String manualmente para enviarla CRUDA.
            
            s_date = target_event['startDate']
            e_date = target_event['endDate']
            
            # URL Literal (f-string) idéntica a la de Bash
            full_url = f"{self.base_url}/metrics/{self.user_id}?type=daily&startDate={s_date}&endDate={e_date}"
            
            # Imprimimos para depurar si falla
            print(f"      DEBUG URL: {full_url}")

            # Usamos session.get pero SIN params=... pasamos la URL entera
            resp_metrics = self.session.get(full_url, timeout=10)
            # ==================================================================
        
            
            if resp_metrics.status_code == 200:
                metrics_data = resp_metrics.json().get('data', [])
                print(metrics_data )
                
                if metrics_data:
                    rows = []
                    cumulative_count = 0
                    
                    # 4. PROCESAR Y CALCULAR ACUMULADO
                    # Aseguramos que los datos estén ordenados por fecha para el acumulado correcto
                    metrics_data.sort(key=lambda x: x.get('date', ''))

                    for item in metrics_data:
                        # 1. La fecha está en la raíz
                        fecha = item.get('date')
                        
                        # 2. El conteo está ANIDADO dentro de 'data'
                        # Ejemplo: item['data']['count']
                        nested_data = item.get('data', {})
                        posts = nested_data.get('count')
                        
                        # Intentamos sacar el acumulado real de la API si existe, sino lo calculamos
                        cumulative = nested_data.get('cumulative')

                        if fecha and posts is not None:
                            rows.append({
                                'Date/Time': fecha,
                                'Posts': int(posts),
                                # Si la API no da acumulado, lo rellenamos luego con pandas (cumsum)
                                'Cumulative_API': cumulative 
                            })
                    # ----------------------------
                    
                    if rows:
                        clean_title = title.replace(" ", "_").replace("?", "").replace("#", "")[:50]
                        filename = f"auto_xtracker_{clean_title}.csv"
                        filepath = os.path.join(self.data_dir, filename)
                        
                        df = pd.DataFrame(rows)
                        
                        # Si la API traía acumulado, usamos ese. Si no, calculamos.
                        if 'Cumulative_API' in df.columns and df['Cumulative_API'].notna().all():
                            df['Cumulative'] = df['Cumulative_API']
                        else:
                            df['Cumulative'] = df['Posts'].cumsum()
                            
                        # Guardar formato final limpio
                        df = df[['Date/Time', 'Posts', 'Cumulative']]
                        df.to_csv(filepath, index=False)
                        
                        print(f"   ✅ Guardado: {filename}")
                        print(f"   📊 Registros: {len(df)} días.")
                    else:
                        print("   ⚠️ JSON recibido pero sin datos válidos tras procesar.")
                else:
                    print(f"   ⚠️ API devolvió lista vacía: {metrics_data}")
            else:
                print(f"   ❌ Error Metrics: {resp_metrics.status_code}")

        except Exception as e:
            print(f"\n   ⚠️ Error: {e}")

# ==========================================
# PRUEBA STANDALONE
# ==========================================
if __name__ == "__main__":
    # Carpeta de prueba
    test_dir = "historic_data_dirty"
    print("--- INICIANDO AUTO-DOWNLOADER ---")
    downloader = XTrackerDownloader(test_dir)
    downloader.sync_history()
    print("--- FIN ---")