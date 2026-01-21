import os
import shutil

# Configuración de carpetas (Las mismas que usa tu bot)
DIRS_TO_CLEAN = [
    "logs",
    "logs/market_tape",
    "logs/snapshots"
]

FILES_TO_DELETE = [
    "trade_history.csv",  # Borra esto para resetear PnL a $0 y Cash a $1000
    "logs/bot_monitor.log",
    "logs/live_history.json" # Vital borrar esto para reiniciar el cerebro Hawkes
]

def clean_slate():
    print("🗑️  INICIANDO LIMPIEZA DE SISTEMA ELON-BOT...")
    
    # 1. Limpiar carpetas
    for folder in DIRS_TO_CLEAN:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        print(f"   Deleted: {filename}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"   Deleted Folder: {filename}")
                except Exception as e:
                    print(f"   ❌ Error borrando {file_path}: {e}")
        else:
            print(f"   Creando carpeta limpia: {folder}")
            os.makedirs(folder, exist_ok=True)

    # 2. Borrar archivos raíz específicos
    for f in FILES_TO_DELETE:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"   Deleted File: {f}")
            except Exception as e:
                print(f"   ❌ Error borrando {f}: {e}")

    print("\n✨ SISTEMA LIMPIO. LISTO PARA V9.2 ✨")
    print("   Saldo inicial: $1000.00")
    print("   Memoria Hawkes: Vacía")
    print("   Market Tape: 00:00:00")

if __name__ == "__main__":
    confirm = input("⚠️  ¿Seguro que quieres borrar TODO el historial y empezar de 0? (s/n): ")
    if confirm.lower() == 's':
        clean_slate()
    else:
        print("Operación cancelada.")