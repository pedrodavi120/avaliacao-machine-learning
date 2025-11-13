import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar o pipeline treinado (deve estar no mesmo diretório)
try:
    pipeline = joblib.load('loan_pipeline.joblib')
    st.success("Modelo de pipeline carregado com sucesso!")
except FileNotFoundError:
    st.error("Arquivo 'loan_pipeline.joblib' não encontrado. Por favor, execute o notebook 'tarefa_2_treinamento.ipynb' primeiro.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

st.title("🏦 Aplicação de Análise de Liberação de Empréstimo")
st.markdown("Preencha os dados abaixo para obter uma predição (Sim/Não) e a probabilidade.")

# --- Criar o formulário com os controles da Instrução 2j ---
# Os valores (ex: 'Male', 'Yes', '0') devem corresponder aos dados originais
# antes do pré-processamento, pois o pipeline cuidará da transformação.

with st.form("loan_form"):
    st.subheader("Informações do Solicitante")
    
    # Sexo (radiobutton)
    gender = st.radio("Sexo", ('Male', 'Female'), index=0)
    
    # Casado (radio)
    married = st.radio("Casado(a)?", ('Yes', 'No'), index=0)
    
    # Dependentes (option/selection)
    dependents = st.selectbox("Número de Dependentes", ('0', '1', '2', '3+'), index=0)
    
    # Educação (radio/selection)
    education = st.radio("Educação", ('Graduate', 'Not Graduate'), index=0)
    
    # Autônomo (radio)
    self_employed = st.radio("Trabalha por Conta Própria (Autônomo)?", ('No', 'Yes'), index=0)
    
    st.subheader("Informações Financeiras")
    
    # Rendimento (text) - Usando number_input para melhor controle
    applicant_income = st.number_input("Rendimento do Solicitante (mensal)", min_value=0, value=5000)
    
    # Valoremprestimo (text) - Usando number_input
    # O pipeline trata NaNs, então podemos permitir 0 ou um valor padrão
    loan_amount = st.number_input("Valor do Empréstimo (em milhares)", min_value=0, value=150)

    # Botão de envio
    submitted = st.form_submit_button("Analisar")

# --- Processamento e Exibição dos Resultados (Instrução 2k) ---
if submitted:
    # Criar um DataFrame com os dados de entrada
    # As colunas devem corresponder EXATAMENTE às usadas no treino do pipeline
    input_data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'LoanAmount': [loan_amount]
    }
    
    # Converter para DataFrame
    input_df = pd.DataFrame(input_data)
    
    st.subheader("Dados Fornecidos:")
    st.dataframe(input_df)
    
    try:
        # Fazer a predição
        prediction = pipeline.predict(input_df)
        
        # Obter as probabilidades (Instrução 2k)
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)
            
            # Resultado (Sim/Não)
            prediction_status = 'Sim' if prediction[0] == 1 else 'Não'
            
            # Probabilidade da classe prevista
            probability_of_prediction = proba[0][prediction[0]]
            
            st.subheader("Resultado da Análise:")
            if prediction_status == 'Sim':
                st.success(f"Status: **{prediction_status}** (Empréstimo Aprovado)")
            else:
                st.error(f"Status: **{prediction_status}** (Empréstimo Negado)")
                
            st.info(f"Probabilidade da decisão: **{probability_of_prediction*100:.2f}%**")
            
            # Exibir probabilidades de ambas as classes
            st.write("Probabilidades Detalhadas:")
            st.write(f"  - Probabilidade (Não): {proba[0][0]*100:.2f}%")
            st.write(f"  - Probabilidade (Sim): {proba[0][1]*100:.2f}%")
            
        else:
            # Fallback caso o modelo não tenha predict_proba
            prediction_status = 'Sim' if prediction[0] == 1 else 'Não'
            st.subheader("Resultado da Análise:")
            st.success(f"Status: **{prediction_status}**")
            st.warning("Este modelo não fornece probabilidades.")

    except Exception as e:
        st.error(f"Ocorreu um erro durante a predição: {e}")