import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar o pipeline treinado (deve estar no mesmo diretório)
try:
    pipeline = joblib.load('loan_pipeline.joblib')
    st.success("Modelo de pipeline carregado com sucesso!")
except FileNotFoundError:
    st.error("Arquivo 'loan_pipeline.joblib' não encontrado. Por favor, execute o notebook 'tarefa_2_corrigido.ipynb' primeiro.")
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sexo (radiobutton)
        gender = st.radio("Sexo", ('Male', 'Female'), index=0)
        
        # Casado (radio)
        married = st.radio("Casado(a)?", ('Yes', 'No'), index=0)
        
        # Dependentes (selectbox)
        dependents = st.selectbox("Dependentes", ('0', '1', '2', '3+'), index=0)

    with col2:
        # Educação (radio)
        education = st.radio("Educação", ('Graduate', 'Not Graduate'), index=0)
        
        # Autônomo (radio)
        self_employed = st.radio("Autônomo?", ('Yes', 'No'), index=1)
        
        # --- CAMPO ADICIONADO ---
        property_area = st.selectbox("Área da Propriedade", ('Urban', 'Semiurban', 'Rural'), index=0)

    
    st.subheader("Informações Financeiras")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Renda (number_input)
        applicant_income = st.number_input("Renda do Solicitante (Mensal)", min_value=0, value=5000)
        
        # Valor do Empréstimo (number_input)
        loan_amount = st.number_input("Valor do Empréstimo (Total)", min_value=0, value=150)
        
        # --- CAMPO ADICIONADO ---
        # (O pipeline foi treinado com 1.0 e 0.0)
        credit_history = st.radio("Possui Histórico de Crédito?", (1.0, 0.0), 
                                  format_func=lambda x: 'Sim' if x == 1.0 else 'Não', index=0)
        
    with col4:
        # --- CAMPO ADICIONADO ---
        coapplicant_income = st.number_input("Renda do Co-solicitante (Mensal)", min_value=0, value=0)
        
        # --- CAMPO ADICIONADO ---
        loan_amount_term = st.number_input("Prazo do Empréstimo (Meses)", min_value=12, value=360, step=12)
    
    
    # --- Botão de Envio ---
    submitted = st.form_submit_button("Analisar")

if submitted:
    try:
        # --- Criar o DataFrame para o pipeline ---
        # A ESTRUTURA DEVE SER IDÊNTICA AO X_train ORIGINAL
        
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'LoanAmount': [loan_amount],
            
            # --- COLUNAS ADICIONADAS AO DICIONÁRIO ---
            'CoapplicantIncome': [coapplicant_income],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        }
        
        # Criar o DataFrame
        # Garantir a ordem das colunas (embora o ColumnTransformer lide com isso, é uma boa prática)
        col_order = [
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', # Numéricas
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', # Categóricas
            'Credit_History' # Binária
        ]
        
        # Filtra o dicionário para o caso de alguma coluna faltar (embora não deva)
        input_df_data = {key: input_data[key] for key in col_order if key in input_data}
        input_df = pd.DataFrame(input_df_data)

        # Obter a predição (Instrução 2k)
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
            st.success(f"Status: **{prediction_status}** (Valor: {prediction[0]})")

    except Exception as e:
        st.error(f"Ocorreu um erro durante a predição: {e}")