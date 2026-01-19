import google.generativeai as genai
from google.api_core import exceptions
import time
import config

class GeminiRotator:
    """
    Gerenciador robusto de rotação de chaves API.
    Implementa retries automáticos e troca de chave em caso de Exaustão de Cota (429).
    """
    def __init__(self):
        self.keys = [k for k in config.GEMINI_KEYS_POOL if "SUA_CHAVE" not in k]
        self.current_index = 0
        self.model_name = 'gemini-2.5-flash' # Modelo rápido e eficiente
        
        if not self.keys:
            raise ValueError("Nenhuma chave API válida encontrada no config.py!")

    def _get_current_key(self):
        return self.keys[self.current_index]

    def _rotate_key(self):
        """Avança para a próxima chave na lista (Circular)."""
        prev_key = self.current_index
        self.current_index = (self.current_index + 1) % len(self.keys)
        print(f"[!] Limite de API atingido na chave {prev_key+1}. Alternando para chave {self.current_index+1}...")
        time.sleep(1) # Pequena pausa para evitar race conditions

    def generate_content(self, prompt, retries_max=6):
        """
        Tenta gerar conteúdo. Se falhar por limite, troca a chave e tenta de novo.
        retries_max = 6 garante que ele possa rodar a lista de 3 chaves duas vezes antes de desistir.
        Erro 403 (chave comprometida) também rotaciona.
        """
        attempts = 0
        last_error = None
        keys_tried = set()

        while attempts < retries_max:
            try:
                current_key = self._get_current_key()
                keys_tried.add(self.current_index)
                
                # Configura com a chave atual
                genai.configure(api_key=current_key)
                model = genai.GenerativeModel(self.model_name)
                
                # Chamada da API
                response = model.generate_content(prompt)
                return response.text

            except exceptions.ResourceExhausted:
                # Erro 429: Cota excedida -> Rotação
                self._rotate_key()
                attempts += 1
            
            except exceptions.PermissionDenied as e:
                # Erro 403: Chave comprometida/inválida -> Rotação
                if "leaked" in str(e).lower() or "invalid" in str(e).lower():
                    print(f"[!] Chave {self.current_index+1} foi bloqueada (403: {str(e)[:50]}...). Alternando...")
                    self._rotate_key()
                    attempts += 1
                else:
                    print(f"[X] Erro de permissão (não é chave comprometida): {e}")
                    last_error = e
                    raise e
            
            except Exception as e:
                # Erros fatais (autenticação inválida, internet off)
                print(f"[X] Erro na API: {e}")
                last_error = e
                raise e

        # Se saiu do while, falhou em todas as tentativas
        if len(keys_tried) == len(self.keys):
            print(f"[!!!] TODAS as {len(self.keys)} chaves API falharam!")
            print(f"Último erro: {last_error}")
            print("\n[!] AÇÃO NECESSÁRIA: Regenere as chaves no Google Cloud Console")
            print("As chaves foram marcadas como 'leaked' e precisam ser substituídas")
        
        raise RuntimeError(f"Todas as {len(self.keys)} chaves API falharam. Regenere em: https://console.cloud.google.com")