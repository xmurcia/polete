import os
import json
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds


class PolyAuth:
    def __init__(self):
        load_dotenv()

        self.private_key = os.getenv("PRIVATE_KEY")
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        self.chain_id = int(os.getenv("CHAIN_ID", "137"))
        self.host = os.getenv("CLOB_HOST", "https://clob.polymarket.com")

        if not self.private_key:
            raise ValueError("❌ PRIVATE_KEY not found in .env file")

        if not self.wallet_address:
            raise ValueError("❌ WALLET_ADDRESS not found in .env file")

        if not self.private_key.startswith("0x"):
            self.private_key = f"0x{self.private_key}"

        self.credentials_path = Path(__file__).parent / ".credentials.json"
        self.client: Optional[ClobClient] = None

        print(f"[PolyAuth] Initialized")
        print(f"  Chain ID: {self.chain_id}")
        print(f"  Host: {self.host}")
        print(f"  Wallet: {self.wallet_address}")

    def _load_cached_credentials(self) -> Optional[ApiCreds]:
        if not self.credentials_path.exists():
            print("[PolyAuth] No cached credentials found")
            return None

        try:
            with open(self.credentials_path, 'r') as f:
                data = json.load(f)

            cached_time = data.get("timestamp", 0)
            age_hours = (time.time() - cached_time) / 3600

            if age_hours >= 24:
                print(f"[PolyAuth] Cached credentials expired ({age_hours:.1f}h old)")
                return None

            creds = ApiCreds(
                api_key=data["api_key"],
                api_secret=data["api_secret"],
                api_passphrase=data["api_passphrase"]
            )

            print(f"[PolyAuth] ✅ Loaded cached credentials ({age_hours:.1f}h old)")
            return creds

        except Exception as e:
            print(f"[PolyAuth] ⚠️  Failed to load cached credentials: {e}")
            return None

    def _save_credentials(self, creds: ApiCreds) -> None:
        try:
            data = {
                "api_key": creds.api_key,
                "api_secret": creds.api_secret,
                "api_passphrase": creds.api_passphrase,
                "timestamp": time.time()
            }

            with open(self.credentials_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[PolyAuth] ✅ Credentials cached to {self.credentials_path.name}")

        except Exception as e:
            print(f"[PolyAuth] ⚠️  Failed to save credentials: {e}")

    def get_client(self) -> ClobClient:
        if self.client is not None:
            return self.client

        try:
            print("[PolyAuth] Step 1: Checking for cached credentials...")
            cached_creds = self._load_cached_credentials()

            if cached_creds:
                print("[PolyAuth] Step 2: Creating L2 client with cached credentials...")
                self.client = ClobClient(
                    host=self.host,
                    key=self.private_key,
                    chain_id=self.chain_id,
                    creds=cached_creds,
                    signature_type=2,
                    funder=self.wallet_address
                )
                print("[PolyAuth] ✅ Client initialized with cached credentials")
                return self.client

            print("[PolyAuth] Step 2: Creating L1 client for credential derivation...")
            client_l1 = ClobClient(
                host=self.host,
                key=self.private_key,
                chain_id=self.chain_id
            )

            print("[PolyAuth] Step 3: Deriving API credentials (signing L1 → L2)...")
            api_creds = client_l1.derive_api_key()
            print("[PolyAuth] ✅ API credentials derived")

            self._save_credentials(api_creds)

            print("[PolyAuth] Step 4: Creating L2 client with new credentials...")
            self.client = ClobClient(
                host=self.host,
                key=self.private_key,
                chain_id=self.chain_id,
                creds=api_creds,
                signature_type=2,
                funder=self.wallet_address
            )

            print("[PolyAuth] ✅ Client initialized successfully")
            return self.client

        except ValueError as e:
            error_msg = str(e).lower()
            if "private" in error_msg or "key" in error_msg:
                raise ValueError(
                    f"❌ Invalid private key format. Ensure PRIVATE_KEY in .env is a valid hex string.\n"
                    f"   Error: {e}"
                )
            raise

        except ConnectionError as e:
            raise ConnectionError(
                f"❌ Network error connecting to {self.host}\n"
                f"   Error: {e}\n"
                f"   Check your internet connection and try again."
            )

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise PermissionError(
                    f"❌ Authentication failed. Check that:\n"
                    f"   1. PRIVATE_KEY matches WALLET_ADDRESS\n"
                    f"   2. Wallet has proper permissions on Polymarket\n"
                    f"   Error: {e}"
                )

            raise RuntimeError(
                f"❌ Failed to initialize client: {e}\n"
                f"   Check your .env configuration"
            )

    def test_connection(self) -> bool:
        try:
            print("\n[PolyAuth] Testing connection...")
            client = self.get_client()

            print("[PolyAuth] Calling get_ok()...")
            result = client.get_ok()

            print(f"[PolyAuth] ✅ Connection successful!")
            print(f"[PolyAuth] Server response: {result}")

            return True

        except Exception as e:
            print(f"[PolyAuth] ❌ Connection test failed: {e}")
            return False

    def get_wallet_address(self) -> str:
        return self.wallet_address
