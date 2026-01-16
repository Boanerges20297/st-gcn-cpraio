import pandas as pd
import config
import os
# Importa nosso novo cliente robusto
from gemini_client import GeminiRotator

def generate_report():
    print("[-] Iniciando Consultoria Tática com IA (Sistema Multi-Chave)...")
    
    if not config.PREDICTION_CSV.exists():
        print(f"[X] Erro: Arquivo de previsão não encontrado.")
        return

    try:
        # 1. Instancia o Rotacionador
        # Se não houver chaves válidas, ele já avisa aqui.
        ai_client = GeminiRotator()
        
        # 2. Preparar os Dados
        df = pd.read_csv(config.PREDICTION_CSV)
        target_col = 'CVLI' if 'CVLI' in df.columns else df.columns[1]
        
        # Filtrar apenas dados relevantes (>0) e Top 5
        df_risk = df[df[target_col] > 0.01].sort_values(by=target_col, ascending=False).head(5)
        
        if df_risk.empty:
            print("[!] Nenhum risco relevante detectado para relatório.")
            return

        data_summary = df_risk.to_string(index=False)
        
        # 3. Prompt Tático (Otimizado)
        prompt = f"""
        CONTEXTO: Você é um Conselheiro de Inteligência Artificial do CPRAIO (Ceará).
        Sua função é ANALISAR e INTERPRETAR dados geográficos e estatísticos, não dar ordens operacionais.
        
        DATA: Próxima Quinzena
        FOCO: Análise interpretativa de risco baseada em padrões espaciais
        
        INPUT - ÍNDICES DE RISCO CRIMINAL POR BAIRRO (Mapa de Calor ST-GCN):
        {data_summary}
        
        TAREFA: Fornecer análise de inteligência sobre os padrões identificados.
        
        ESTRUTURA REQUERIDA:
        
        1. **INTERPRETAÇÃO DO MAPA**: 
           - Descreva o que o mapa de calor mostra em termos de concentração espacial
           - Identifique corredores, clusters ou padrões anômalos
           - Analise distribuição geográfica (periférico vs central)
        
        2. **ANÁLISE DE CONVERGÊNCIA**:
           - Quais bairros concentram maior risco?
           - Existem padrões de progressão ou expansão?
           - Há sobreposição com áreas críticas históricas?
        
        3. **FATORES DE RISCO IDENTIFICÁVEIS**:
           - Que características geográficas podem explicar os picos?
           - Há correlações espaciais entre pontos quentes?
           - Quais áreas sofrem pressão contínua?
        
        4. **RECOMENDAÇÕES ANALÍTICAS** (não operacionais):
           - Que áreas demandam aprofundamento de investigação?
           - Quais padrões requerem monitoramento contínuo?
           - Que dados adicionais poderiam esclarecer os achados?
        
        Tom: Inteligência, Analítico, Conselheiro Especializado.
        Evitar: Ordens militares, Termos táticos, Linguagem de comando.
        Foco: Evidências, Padrões, Recomendações de análise.
        """
        
        # 4. Chamada Robusta (O Rotacionador cuida dos erros)
        print("[-] Enviando solicitação ao cluster Gemini...")
        report_text = ai_client.generate_content(prompt)
        
        # 5. Exibir e Salvar
        print("\n" + "="*60)
        print(report_text)
        print("="*60 + "\n")
        
        report_path = config.REPORT_DIR / "boletim_inteligencia.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[V] Boletim salvo em: {report_path}")

    except ValueError as ve:
        print(f"[!] Configuração: {ve}")
    except RuntimeError as re:
        print(f"[X] Falha no Cluster API: {re}")
    except Exception as e:
        print(f"[X] Erro inesperado: {e}")

if __name__ == "__main__":
    generate_report()