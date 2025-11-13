Para executar esta avaliação, siga os passos:

**Dependências Comuns:**
Certifique-se de ter as bibliotecas Python necessárias instaladas:
pip install jupyter pandas numpy matplotlib scikit-learn xgboost joblib streamlit imbalanced-learn

**Tarefa 1: MNIST**
1.  Execute o notebook `tarefa_1_mnist.ipynb`.
2.  Isso levará algum tempo, especialmente nas etapas '1a' (cross-validation) e '1e' (RandomizedSearch), pois os modelos SVC e KNN são lentos no dataset MNIST completo.
3.  Ao final, os arquivos `best_mnist_model.joblib` (o modelo treinado) e `mnist_scaler.joblib` (o scaler usado) serão salvos no mesmo diretório.

**Tarefa 2: Análise de Empréstimo**
1.  **Treinamento:**
    * Primeiro, execute o notebook `tarefa_2_treinamento.ipynb`.
    * Este notebook irá baixar o `loan.csv`, processar os dados, treinar 5 pipelines com GridSearchCV e salvar o melhor pipeline como `loan_pipeline.joblib`.
2.  **Aplicação Web (Streamlit):**
    * Após a criação do `loan_pipeline.joblib`, abra seu terminal ou prompt de comando.
    * Navegue até o diretório onde os arquivos estão salvos.
    * Execute o seguinte comando para iniciar o aplicativo Streamlit:
        ```bash
        streamlit run tarefa_2_app.py
        ```
    * Abra o endereço local (ex: http://localhost:8501) que aparece no seu terminal para interagir com a aplicação.